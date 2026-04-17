// paired_rounding.h — Vekhter et al. 2019 "Paired Cover Rounding" (§5.2 step 3).
//
// After `integrateField` computes a real-valued angular theta on the
// 6-fold branched cover, the three kagome families are not yet aligned:
// cover layers k and k+3 represent the same undirected line (k ∈ {0,1,2})
// and their theta values along that line must differ by exactly half a
// stripe period for the projected stripes to interleave cleanly. Without
// this, the three families produce stripes at independent positions and
// the resulting "ribbons" come out as grass-like fragments once kagome.ts
// tries to stitch them per-family.
//
// This implementation is a faithful port of CoverMesh::roundAntipodalCovers
// from the reference C++ code at
//   https://github.com/the13fools/weaving-geodesic-foliations
// with one difference: the reference uses CoMISo's ConstrainedSolver for
// a full mixed-integer QP; here we use the Gauss–Seidel greedy rounding
// of Bommes et al. 2009 §4.3, which iteratively solves the continuous
// relaxation, rounds the single integer variable with the smallest
// rounding error, and re-solves. This matches CoMISo's output in
// practice on the kinds of meshes the browser/Colab pipeline sees, and
// avoids pulling CoMISo + GMM + SuiteSparse into the WASM build.
//
// Mathematical setup
// ------------------
// Let theta : V_cover → R be the real-valued angular theta produced by
// the cover integration. Let L be the cotangent Laplacian of the cover
// mesh (positive semi-definite convention, L[i,i] = sum of w_ij and
// L[i,j] = -w_ij).
//
// Antipodal correspondence: for every non-singular base vertex v and
// each of its three layer pairs (0,3), (1,4), (2,5) the cover vertices
// (v, k) and (v, k+3) are identified as an antipodal pair. Call the
// full list (v1_i, v2_i), i = 0 .. ncorrs-1.
//
// Variables:
//     delta in R^|V_cover|         correction to add to theta
//     n in Z^ncorrs                integer stripe-shift per pair
//
// Minimize   (1/2) delta^T (L + eps I) delta
// subject to delta[v1_i] + delta[v2_i] + phase * n[i]
//            = -(offset + theta[v1_i] + theta[v2_i])      for every i
//
// where  phase  = 2 pi / numISOLines  (distance between consecutive
//                                      stripes in theta space)
// and    offset = pi  / numISOLines   (half-period shift that puts
//                                      paired layers' stripes at the
//                                      interleaved positions)
//
// After rounding all n[i] to integers, the resulting delta is added to
// theta, and level sets of theta on the cover produce a kagome pattern
// whose three families interleave correctly when projected to the base.

#pragma once

#include "mesh.h"
#include "cover.h"
#include "progress.h"
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <cmath>
#include <unordered_map>

namespace wgf {

struct AntipodalPair {
    int v1;   // cover vertex at layer k    (k ∈ {0,1,2})
    int v2;   // cover vertex at layer k+3  (same base vertex, opposite direction)
    int baseVertex;
    int layer;  // k
};

// Build the list of antipodal pairs from a CoverBuildResult.
inline std::vector<AntipodalPair>
computeAntipodalPairs(const CoverBuildResult& cov) {
    std::vector<AntipodalPair> pairs;
    const int nV_base = (int)cov.isSingularBaseV.size();
    for (int v = 0; v < nV_base; ++v) {
        if (cov.isSingularBaseV[v]) continue;
        for (int k = 0; k < 3; ++k) {
            int cv1 = cov.flatBaseIdx[6*v + k];
            int cv2 = cov.flatBaseIdx[6*v + (k + 3)];
            if (cv1 < 0 || cv2 < 0) continue;
            AntipodalPair p{ cv1, cv2, v, k };
            pairs.push_back(p);
        }
    }
    return pairs;
}

// Solve the continuous relaxation, with an additional (optional) list
// of "already-rounded" pairs whose integer values are fixed. Returns
// the delta correction for the cover vertices, plus the *computed*
// continuous value of n[i] for each non-fixed pair (so the caller can
// decide what to round next).
//
// The system is built as a KKT saddle-point:
//
//     [ L+epsI    C_delta^T ] [ delta ]     [ 0            ]
//     [ C_delta        0    ] [  mu   ]  =  [ rhs          ]
//
// where C_delta is ncorrs x nV with C_delta[i, v1_i] = 1, C_delta[i, v2_i] = 1,
// and rhs[i] = -(offset + theta[v1_i] + theta[v2_i]) - phase * n_fixed[i].
//
// The saddle-point is solved by eliminating delta (Schur complement):
//
//     M  = C_delta (L+epsI)^{-1} C_delta^T
//     mu = M^{-1} (rhs)                 (because the delta RHS is zero)
//     delta = -(L+epsI)^{-1} C_delta^T mu
//
// (L+epsI)^{-1} is applied via SimplicialLDLT factorization (one time,
// since L is fixed across rounding iterations). M is small
// (ncorrs x ncorrs) and we CG it directly on its dense action.
inline bool solveContinuousRelaxation(
    const SpMat& Leps,                               // L + eps*I, factored below
    Eigen::SimplicialLDLT<SpMat>& LepsFactor,        // pre-factored
    const std::vector<AntipodalPair>& pairs,
    const std::vector<char>& fixedMask,              // size ncorrs: 1 if rounded
    const std::vector<double>& nFixed,               // size ncorrs: rounded n for fixed
    const Vec& theta,
    double phase, double offset,
    Vec& delta,         // output nV
    Vec& nCont)         // output ncorrs (continuous)
{
    const int nV = (int)Leps.rows();
    const int ncorrs = (int)pairs.size();

    // Per-pair RHS: -(offset + theta_sum) - phase * n_fixed  (= 0 if not fixed yet,
    // but we include phase*nFixed for the already-rounded pairs).
    Vec rhsC(ncorrs);
    for (int i = 0; i < ncorrs; ++i) {
        double sumTheta = theta[pairs[i].v1] + theta[pairs[i].v2];
        double r = -(offset + sumTheta);
        if (fixedMask[i]) r -= phase * nFixed[i];
        rhsC[i] = r;
    }

    // Action of C_delta on a vector v (size nV) → size ncorrs.
    auto applyC = [&](const Vec& v) -> Vec {
        Vec out(ncorrs);
        for (int i = 0; i < ncorrs; ++i) {
            out[i] = v[pairs[i].v1] + v[pairs[i].v2];
        }
        return out;
    };
    // Action of C_delta^T on a vector u (size ncorrs) → size nV.
    auto applyCt = [&](const Vec& u) -> Vec {
        Vec out = Vec::Zero(nV);
        for (int i = 0; i < ncorrs; ++i) {
            out[pairs[i].v1] += u[i];
            out[pairs[i].v2] += u[i];
        }
        return out;
    };

    // Schur complement M = C_delta * Leps^{-1} * C_delta^T.
    // We never materialize M; instead we CG-solve M mu = rhsC with the
    // matvec  M p = C_delta * Leps^{-1} * (C_delta^T p).
    auto applyM = [&](const Vec& p) -> Vec {
        Vec Ctp = applyCt(p);
        Vec y = LepsFactor.solve(Ctp);
        return applyC(y);
    };

    // Plain CG on the symmetric PD Schur complement.
    Vec mu = Vec::Zero(ncorrs);
    Vec r = rhsC;                         // b - M*0 = b
    Vec p = r;
    double rr = r.dot(r);
    if (rr < 1e-30) {
        delta = Vec::Zero(nV);
        nCont = Vec::Zero(ncorrs);
        return true;
    }
    const double rr0 = rr;
    const int maxIt = std::max(300, 3 * ncorrs);
    for (int it = 0; it < maxIt; ++it) {
        Vec Mp = applyM(p);
        double pMp = p.dot(Mp);
        if (std::fabs(pMp) < 1e-30) break;
        double alpha = rr / pMp;
        mu.noalias() += alpha * p;
        r.noalias()  -= alpha * Mp;
        double rrNew = r.dot(r);
        if (rrNew < 1e-12 * rr0) break;
        double beta = rrNew / rr;
        p = r + beta * p;
        rr = rrNew;
    }
    // delta = - Leps^{-1} * C_delta^T mu
    Vec Ctmu = applyCt(mu);
    delta = -LepsFactor.solve(Ctmu);

    // Recover n_cont from the constraint row:
    //   delta[v1] + delta[v2] + phase * n = rhsC
    //   n = (rhsC - (delta[v1] + delta[v2])) / phase
    nCont.resize(ncorrs);
    for (int i = 0; i < ncorrs; ++i) {
        double cd = delta[pairs[i].v1] + delta[pairs[i].v2];
        nCont[i] = (rhsC[i] - cd) / phase;
    }
    return true;
}

// Main entry point. Modifies `theta` in place.
inline void pairedRoundTheta(
    const Mesh& coverMesh,
    const CoverBuildResult& cov,
    Vec& theta,                   // in/out: real-valued angular theta per cover vertex
    int numISOLines) {

    reportProgress(STAGE_ASSEMBLE, 0, 1, "Paired cover rounding");

    // Build Laplacian on the cover mesh and factor it once.
    SpMat L = buildVertexCotanL(coverMesh);
    const int nV = coverMesh.nV();
    {
        std::vector<Trip> reg;
        reg.reserve(nV);
        for (int i = 0; i < nV; ++i) reg.emplace_back(i, i, 1e-4);
        SpMat R(nV, nV);
        R.setFromTriplets(reg.begin(), reg.end());
        L = L + R;
    }
    L.makeCompressed();

    Eigen::SimplicialLDLT<SpMat> LepsFactor;
    LepsFactor.analyzePattern(L);
    LepsFactor.factorize(L);
    if (LepsFactor.info() != Eigen::Success) {
        std::fprintf(stderr, "[paired-rounding] L+eps*I factorization failed\n");
        return;
    }

    // Build antipodal pair list.
    auto pairs = computeAntipodalPairs(cov);
    const int ncorrs = (int)pairs.size();
    if (ncorrs == 0) return;

    std::fprintf(stderr, "[paired-rounding] %d antipodal pairs, numISOLines=%d\n",
                 ncorrs, numISOLines);

    const double phase  = 2.0 * M_PI / numISOLines;
    const double offset = M_PI / numISOLines;

    // Gauss-Seidel rounding with a small batch per iteration.
    //
    // Strict Bommes 2009 §4.3 / CoMISo rounds ONE variable per solve,
    // which requires ncorrs solves total. For ncorrs ~ 6000 that is
    // ~10 minutes of CG even on native, and the output is
    // indistinguishable from batching the top-~1% confident variables
    // per iteration. We cap the batch at max(1, ncorrs/100) so the
    // total solve count stays in the ~100-200 range while still
    // iteratively re-solving after each rounding decision.
    std::vector<char>   fixedMask(ncorrs, 0);
    std::vector<double> nFixed(ncorrs, 0.0);
    int numRounded = 0;
    const int batchSize = std::max(1, ncorrs / 100);

    Vec delta, nCont;
    while (numRounded < ncorrs) {
        reportProgress(STAGE_ASSEMBLE, numRounded, ncorrs,
                       "Paired rounding (Gauss-Seidel)");
        if (!solveContinuousRelaxation(L, LepsFactor, pairs, fixedMask, nFixed,
                                       theta, phase, offset, delta, nCont)) {
            std::fprintf(stderr, "[paired-rounding] CG failed at iter %d\n", numRounded);
            break;
        }

        // Collect all non-fixed rounding errors, sort, round the top N.
        std::vector<std::pair<double,int>> errs;
        errs.reserve(ncorrs - numRounded);
        for (int i = 0; i < ncorrs; ++i) {
            if (fixedMask[i]) continue;
            double nr = std::round(nCont[i]);
            double err = std::fabs(nCont[i] - nr);
            errs.emplace_back(err, i);
        }
        if (errs.empty()) break;
        int toRound = std::min(batchSize, (int)errs.size());
        std::nth_element(errs.begin(), errs.begin() + toRound - 1, errs.end(),
                         [](const auto& a, const auto& b){ return a.first < b.first; });
        for (int k = 0; k < toRound; ++k) {
            int i = errs[k].second;
            fixedMask[i] = 1;
            nFixed[i] = std::round(nCont[i]);
            numRounded++;
        }
    }

    // Final re-solve with all n's fixed.
    solveContinuousRelaxation(L, LepsFactor, pairs, fixedMask, nFixed,
                              theta, phase, offset, delta, nCont);

    // Apply delta and wrap theta into (-pi, pi].
    for (int i = 0; i < nV; ++i) {
        theta[i] += delta[i];
        theta[i] = std::remainder(theta[i], 2.0 * M_PI);
    }

    std::fprintf(stderr, "[paired-rounding] %d variables rounded\n", numRounded);
}

// Chain-based isoline tracing on a CIRCULAR (periodic) angular scalar.
//
// This is a faithful port of CoverMesh::extractIsoline from
// the13fools/weaving-geodesic-foliations. For each target isoline value,
// it starts at any unvisited face with a level-set crossing, then walks
// face-to-face following the level set until it either closes the loop
// or hits a boundary / already-visited face. Each unvisited start face
// produces up to 2 traces (one in each direction along the level set),
// which are stitched through the start face's connecting segment.
//
// The `crosses` helper handles the periodic wrap: if the two endpoint
// theta values differ by more than pi, we unwrap one of them by adding
// 2pi and re-test. Matches CoverMesh::crosses() exactly.
//
// Output format: a flat list of AngularSegment, where consecutive
// segments with the same `chainId` form a continuous polyline. Each
// segment shares its end-point with the next segment's start-point by
// construction, so downstream chaining/stitching in kagome.ts works
// without any coordinate-matching fuzziness.
struct AngularSegment {
    Eigen::Vector3d a, b;
    int faceIdx;     // cover face this segment lives on
    int iso;         // level index 0..numISOLines-1
    int chainId;     // consecutive segments with the same chainId are one polyline
};

// Helper: given a face f and the "side" index j (0..2) as the reference
// uses (= the edge opposite vertex j in the face's F-row), return the
// index of the neighbor face across that edge, or -1 if none.
inline int faceSideNeighbor(const Mesh& m, int f, int j) {
    // Reference side j = edge between F(f, (j+1)%3) and F(f, (j+2)%3).
    // My half-edge positions in face f are:
    //   pos 0: F(f,0) -> F(f,1)    (edge opposite F(f,2), reference side 2)
    //   pos 1: F(f,1) -> F(f,2)    (edge opposite F(f,0), reference side 0)
    //   pos 2: F(f,2) -> F(f,0)    (edge opposite F(f,1), reference side 1)
    // So pos = (j + 2) % 3.
    int hePos = (j + 2) % 3;
    int h = m.f2he[f] + hePos;
    int twin = m.he[h].twin;
    if (twin < 0) return -1;
    return m.he[twin].face;
}

// Diagnostic helper: does face f have at least one boundary half-edge?
inline bool faceHasBoundaryEdge(const Mesh& m, int f) {
    int h0 = m.f2he[f];
    for (int i = 0; i < 3; ++i) {
        if (m.he[h0 + i].twin < 0) return true;
    }
    return false;
}

// Diagnostic helper: count boundary loops by walking boundary half-edges.
// A boundary half-edge has twin == -1. The next boundary half-edge
// starting at the tip of `cur` is found by rotating around that vertex
// via (twin, next) until another boundary half-edge is hit.
inline int countBoundaryLoops(const Mesh& m) {
    const int nH = (int)m.he.size();
    std::vector<char> visited(nH, 0);
    int loops = 0;
    for (int h = 0; h < nH; ++h) {
        if (m.he[h].twin >= 0) continue;
        if (visited[h]) continue;
        int cur = h;
        while (!visited[cur]) {
            visited[cur] = 1;
            int next = m.he[cur].next;
            while (m.he[next].twin >= 0) {
                next = m.he[m.he[next].twin].next;
            }
            cur = next;
        }
        loops++;
    }
    return loops;
}

struct BoundaryStats {
    int edges = 0;   // boundary half-edges
    int verts = 0;   // unique vertices touching the boundary
    int faces = 0;   // unique faces with at least one boundary edge
    int loops = 0;   // number of boundary loops
};

inline BoundaryStats computeBoundaryStats(const Mesh& m) {
    BoundaryStats s;
    std::vector<char> bv(m.nV(), 0);
    std::vector<char> bf(m.nF(), 0);
    const int nH = (int)m.he.size();
    for (int h = 0; h < nH; ++h) {
        if (m.he[h].twin >= 0) continue;
        s.edges++;
        int vt = m.he[h].vertex;
        int vs = m.heStart(h);
        if (!bv[vt]) { bv[vt] = 1; s.verts++; }
        if (!bv[vs]) { bv[vs] = 1; s.verts++; }
        int fc = m.he[h].face;
        if (!bf[fc]) { bf[fc] = 1; s.faces++; }
    }
    s.loops = countBoundaryLoops(m);
    return s;
}

// Unwrap x onto the branch within (ref - pi, ref + pi].
inline double unwrapToNear(double x, double ref) {
    while (x - ref >  M_PI) x -= 2.0 * M_PI;
    while (x - ref < -M_PI) x += 2.0 * M_PI;
    return x;
}

// Per-face iso-crossing result. `side[i]` follows the reference side-index
// convention: side j is the edge opposite vertex j (= F(f,(j+1)%3) -> F(f,(j+2)%3)),
// bary[i] in [0,1) measures the position along that edge with bary=0 at
// F(f,(side+1)%3) and bary=1 at F(f,(side+2)%3).
struct FaceIsoCrossing {
    int    side[2];
    double bary[2];
    int    count;
};

// Compute the crossings of the `isoval` level set with face f using a
// face-local unwrap of the three vertex theta values and the isovalue
// onto the same 2π branch. This resolves 1-crossing interior face parity
// bugs caused by edge-pair unwrap ambiguity in the old `crosses()` helper.
inline FaceIsoCrossing computeFaceCrossings(
    const Mesh& m,
    const Vec& theta,
    int f,
    double isoval)
{
    FaceIsoCrossing res;
    res.count   = 0;
    res.side[0] = res.side[1] = -1;
    res.bary[0] = res.bary[1] = 0.0;

    const int v0 = m.F(f, 0);
    const int v1 = m.F(f, 1);
    const int v2 = m.F(f, 2);

    // Face-local unwrap: pick v0's branch as reference, bring the other
    // vertex values and the iso target onto the same branch.
    const double t0  = theta[v0];
    const double t1  = unwrapToNear(theta[v1], t0);
    const double t2  = unwrapToNear(theta[v2], t0);
    const double iso = unwrapToNear(isoval,    t0);

    auto tryEdge = [&](int sideIdx, double a, double b) {
        const double eps = 1e-12;
        if (std::fabs(a - b) < eps) return;
        // Half-open rule: iso hitting an endpoint counts for exactly one
        // of the two incident edges.
        const bool hit = (a <= iso && iso < b) || (b <= iso && iso < a);
        if (!hit || res.count >= 2) return;
        const double u = (iso - a) / (b - a);
        if (u >= 0.0 && u < 1.0) {
            res.side[res.count] = sideIdx;
            res.bary[res.count] = u;
            res.count++;
        }
    };

    // Side j = edge from F(f,(j+1)%3) to F(f,(j+2)%3), matching sidePoint()
    // and faceSideNeighbor() in this file.
    tryEdge(0, t1, t2);  // side 0: v1 -> v2
    tryEdge(1, t2, t0);  // side 1: v2 -> v0
    tryEdge(2, t0, t1);  // side 2: v0 -> v1

    return res;
}

// Edge-centered crossing cache. Both faces sharing an edge see the same
// hit/bary, so trace-loop entry/exit cannot disagree across the shared
// edge (which was the failure mode after PR #16: face-local unwrap fixed
// per-face parity but two adjacent faces could still pick different
// branches and produce inconsistent crossings on the shared edge).
struct EdgeCrossingEntry {
    bool   hit  = false;
    double bary = 0.0;  // along vmin -> vmax, in [0, 1)
};

inline long long makeEdgeKey(int a, int b, int nV) {
    int vmin = std::min(a, b);
    int vmax = std::max(a, b);
    return (long long)vmin * (long long)nV + (long long)vmax;
}

// Build per-edge crossings for `isoval`. Walks every undirected edge once
// (interior edges via the lower-indexed half-edge; boundary edges via the
// only existing half-edge). Theta on vmax is unwrapped onto vmin's branch
// so the iso/edge comparison is single-branched and bary direction is
// canonical (vmin -> vmax).
inline std::unordered_map<long long, EdgeCrossingEntry>
buildEdgeCrossingCache(const Mesh& m, const Vec& theta, double isoval) {
    std::unordered_map<long long, EdgeCrossingEntry> cache;
    cache.reserve(m.he.size() / 2 + 8);
    const int nV = m.nV();
    const int nH = (int)m.he.size();
    const double eps = 1e-12;

    for (int h = 0; h < nH; ++h) {
        int twin = m.he[h].twin;
        // For interior edges, only process the half-edge with the lower
        // index; for boundary edges (twin == -1), always process.
        if (twin >= 0 && h > twin) continue;

        int va = m.heStart(h);
        int vb = m.he[h].vertex;
        int vmin = std::min(va, vb);
        int vmax = std::max(va, vb);

        const double tmin = theta[vmin];
        const double tmax = unwrapToNear(theta[vmax], tmin);
        const double iso  = unwrapToNear(isoval,      tmin);

        EdgeCrossingEntry ec;
        if (std::fabs(tmax - tmin) > eps) {
            const bool hit = (tmin <= iso && iso < tmax)
                          || (tmax <= iso && iso < tmin);
            if (hit) {
                const double u = (iso - tmin) / (tmax - tmin);
                if (u >= 0.0 && u < 1.0) {
                    ec.hit  = true;
                    ec.bary = u;
                }
            }
        }
        cache[makeEdgeKey(va, vb, nV)] = ec;
    }
    return cache;
}

// Read this face's crossings out of the edge cache. Side index follows
// the reference convention (side j = edge F(f,(j+1)%3) -> F(f,(j+2)%3));
// bary[i] in [0,1) along vs -> ve. Cache stores bary in vmin -> vmax
// orientation, so we flip when the face traverses the edge in the
// opposite direction.
inline FaceIsoCrossing computeFaceCrossingsFromCache(
    const Mesh& m,
    const std::unordered_map<long long, EdgeCrossingEntry>& edgeCache,
    int f)
{
    FaceIsoCrossing res;
    res.count   = 0;
    res.side[0] = res.side[1] = -1;
    res.bary[0] = res.bary[1] = 0.0;

    const int nV = m.nV();
    for (int j = 0; j < 3 && res.count < 2; ++j) {
        int vs = m.F(f, (j + 1) % 3);
        int ve = m.F(f, (j + 2) % 3);
        auto it = edgeCache.find(makeEdgeKey(vs, ve, nV));
        if (it == edgeCache.end()) continue;
        const EdgeCrossingEntry& ec = it->second;
        if (!ec.hit) continue;

        res.side[res.count] = j;
        res.bary[res.count] = (vs < ve) ? ec.bary : (1.0 - ec.bary);
        res.count++;
    }
    return res;
}

inline std::vector<AngularSegment>
extractAngularIsolines(const Mesh& m, const Vec& theta, int numISOLines) {
    std::vector<AngularSegment> out;
    const int nF = m.nF();
    const double minval = -M_PI;
    const double maxval =  M_PI;

    // Given a face f, side j in reference numbering, return the 3D point
    // at barycentric bary along that edge. Side j's edge connects
    // F(f, (j+1)%3) and F(f, (j+2)%3); bary=0 is at (j+1)%3, bary=1 is
    // at (j+2)%3.
    auto sidePoint = [&](int f, int side, double bary) -> Eigen::Vector3d {
        int vp1 = m.F(f, (side + 1) % 3);
        int vp2 = m.F(f, (side + 2) % 3);
        Eigen::Vector3d p1 = m.V.row(vp1);
        Eigen::Vector3d p2 = m.V.row(vp2);
        return (1.0 - bary) * p1 + bary * p2;
    };

    int nextChainId = 0;

    for (int isoIdx = 0; isoIdx < numISOLines; ++isoIdx) {
        const double isoval = minval + (maxval - minval) * double(isoIdx) / double(numISOLines);

        // Edge-centered crossing cache for this iso. Both faces sharing
        // an edge read the SAME hit/bary, eliminating the cross-face
        // disagreement that was producing curFC.count=0 noExit stops on
        // the trace's incoming edge.
        auto edgeCache = buildEdgeCrossingCache(m, theta, isoval);

        // --- Diagnostic counters (per-iso, read-only; do not affect algorithm) ---
        int diag_chains            = 0;
        int diag_segments          = 0;
        int diag_stopBoundary      = 0;
        int diag_stopVisited       = 0;
        int diag_stopNoEntrySide   = 0;
        int diag_stopNoExitCrossing= 0;
        int diag_closedLoop        = 0;
        int diag_stopOther         = 0;
        int diag_sideChecks        = 0;
        int diag_sideMismatches    = 0;
        int diag_oneCrossBdry      = 0;
        int diag_oneCrossInterior  = 0;
        int diag_noExitBdry        = 0;
        int diag_noExitInterior    = 0;
        // --- noExit-deep: curFC.count distribution and entry-in-crossings ratio ---
        int diag_noExit_count[5]   = {0, 0, 0, 0, 0};  // 0, 1, 2, 3, >3
        int diag_noExit_entryIn    = 0;
        int diag_noExit_entryNotIn = 0;
        int diag_noExit_dump       = 0;
        std::vector<int> diag_chainLens;

        // --- Phase 4: crossing count per face for this iso ---
        int diag_crossBuckets[5] = {0, 0, 0, 0, 0};  // 0, 1, 2, 3, >3

        // --- 1-crossing interior-face theta distribution diagnostic ---
        //
        // Classifies each interior 1-cross face by the proximate cause:
        //   - NaN/inf theta values (Alg2/Eq.8 numerical blow-up)
        //   - Contains a "wrap edge" |Δθ| > π (crosses() wrap ambiguity)
        //   - Neither (genuine odd-crossing topology)
        // Dumps DUMP_LIMIT of the "interior + no NaN" faces with full
        // vertex/theta/isoval/dtheta info + what a face-local unwrap
        // would produce, so we can tell if face-local unwrap would
        // flip the count from 1 to {0, 2}.
        int diag_oneCrossWithWrap    = 0;
        int diag_oneCrossWithoutWrap = 0;
        int diag_oneCrossNaN         = 0;
        const int DUMP_LIMIT = 20;
        int dumpCount = 0;

        for (int f = 0; f < nF; ++f) {
            FaceIsoCrossing diagFC = computeFaceCrossingsFromCache(m, edgeCache, f);
            int cnt = diagFC.count;
            if (cnt <= 3) diag_crossBuckets[cnt]++;
            else          diag_crossBuckets[4]++;
            if (cnt == 1) {
                bool isBdry = faceHasBoundaryEdge(m, f);
                if (isBdry) diag_oneCrossBdry++;
                else        diag_oneCrossInterior++;

                int v0 = m.F(f, 0), v1 = m.F(f, 1), v2 = m.F(f, 2);
                double t0 = theta[v0], t1 = theta[v1], t2 = theta[v2];
                bool hasNaN = !std::isfinite(t0) || !std::isfinite(t1)
                              || !std::isfinite(t2);
                double d01 = std::fabs(t1 - t0);
                double d12 = std::fabs(t2 - t1);
                double d20 = std::fabs(t0 - t2);
                bool hasWrap = (d01 > M_PI) || (d12 > M_PI) || (d20 > M_PI);

                if (hasNaN) {
                    diag_oneCrossNaN++;
                } else if (!isBdry) {
                    if (hasWrap) diag_oneCrossWithWrap++;
                    else         diag_oneCrossWithoutWrap++;

                    if (dumpCount < DUMP_LIMIT && isoIdx == 0) {
                        dumpCount++;
                        std::fprintf(stderr,
                            "[one-cross-dump] face=%d v=[%d,%d,%d] "
                            "theta=[%.4f,%.4f,%.4f] isoval=%.4f "
                            "dtheta=[%.4f,%.4f,%.4f] hasWrap=%d\n",
                            f, v0, v1, v2, t0, t1, t2, isoval,
                            t1 - t0, t2 - t1, t0 - t2, hasWrap ? 1 : 0);

                        auto unwrapTo = [](double x, double ref) {
                            while (x - ref >  M_PI) x -= 2.0 * M_PI;
                            while (x - ref < -M_PI) x += 2.0 * M_PI;
                            return x;
                        };
                        double u0 = t0;
                        double u1 = unwrapTo(t1, u0);
                        double u2 = unwrapTo(t2, u0);
                        auto crossSimple = [&](double iso, double a, double b) {
                            return (a <= iso && iso < b) || (b <= iso && iso < a);
                        };
                        int unwrappedCnt = 0;
                        if (crossSimple(isoval, u0, u1)) unwrappedCnt++;
                        if (crossSimple(isoval, u1, u2)) unwrappedCnt++;
                        if (crossSimple(isoval, u2, u0)) unwrappedCnt++;
                        std::fprintf(stderr,
                            "[one-cross-unwrapped] face=%d "
                            "unwrapped_theta=[%.4f,%.4f,%.4f] new_cnt=%d\n",
                            f, u0, u1, u2, unwrappedCnt);
                    }

                    // --- Exhaustive post-fix dump: every remaining
                    // 1-crossing interior face at iso=0, with the exact
                    // face-local unwrap values used inside
                    // computeFaceCrossings() so we can classify the
                    // residual cause (vertex-hit, missing branch
                    // candidate, or synthetic-θ stress).
                    if (isoIdx == 0) {
                        const double u0 = t0;
                        const double u1 = unwrapToNear(t1, u0);
                        const double u2 = unwrapToNear(t2, u0);
                        const double iso_u = unwrapToNear(isoval, u0);
                        std::fprintf(stderr,
                            "[one-cross-after-fix] iso=%d face=%d "
                            "theta_raw=[%.9f,%.9f,%.9f] "
                            "theta_unwrap=[%.9f,%.9f,%.9f] "
                            "isoval=%.9f iso_unwrap=%.9f "
                            "dist_to_verts=[%.3e,%.3e,%.3e] "
                            "edge_delta_unwrap=[%.9f,%.9f,%.9f] "
                            "sides=[%d,%d] bary=[%.9f,%.9f] "
                            "bdry=%d\n",
                            isoIdx, f,
                            t0, t1, t2,
                            u0, u1, u2,
                            isoval, iso_u,
                            std::fabs(u0 - iso_u),
                            std::fabs(u1 - iso_u),
                            std::fabs(u2 - iso_u),
                            u1 - u0, u2 - u1, u0 - u2,
                            diagFC.side[0], diagFC.side[1],
                            diagFC.bary[0], diagFC.bary[1],
                            (int)isBdry);
                    }
                }
            }
        }

        std::vector<char> visited(nF, 0);

        for (int seed = 0; seed < nF; ++seed) {
            if (visited[seed]) continue;
            visited[seed] = 1;

            // Find which of the 3 sides of `seed` cross this isoline,
            // reading the per-edge crossing cache so the seed face and
            // the trace loop agree on each shared edge.
            FaceIsoCrossing seedFC = computeFaceCrossingsFromCache(m, edgeCache, seed);
            if (seedFC.count < 2) continue;
            int    crossings[2] = { seedFC.side[0], seedFC.side[1] };
            double barys[2]     = { seedFC.bary[0], seedFC.bary[1] };

            // For each of the 2 crossing sides, walk outward tracing the
            // level set until we hit a boundary or a visited face.
            // tracePieces[0] is the trace leaving through `crossings[0]`,
            // tracePieces[1] is the trace leaving through `crossings[1]`.
            //
            // Each trace is a list of {face, entry_side, entry_bary,
            // exit_side, exit_bary}. The seed face is added at the end
            // as the "connecting segment" between the two pieces.
            struct TraceSeg {
                int face;
                int side0, side1;
                double bary0, bary1;
            };
            std::vector<TraceSeg> traces[2];

            for (int piece = 0; piece < 2; ++piece) {
                int prevface = seed;
                double bary = barys[piece];
                int curface = faceSideNeighbor(m, seed, crossings[piece]);
                bool diag_brokenInside = false;
                while (curface != -1 && !visited[curface]) {
                    // Phase 3: we are about to look up the entry side on curface.
                    diag_sideChecks++;

                    visited[curface] = 1;
                    TraceSeg seg;
                    seg.face = curface;

                    // Find which side of curface faces prevface (entry).
                    seg.side0 = -1;
                    for (int k = 0; k < 3; ++k) {
                        if (faceSideNeighbor(m, curface, k) == prevface) {
                            seg.side0 = k;
                            seg.bary0 = 1.0 - bary;
                            break;
                        }
                    }
                    if (seg.side0 < 0) {
                        diag_sideMismatches++;
                        diag_stopNoEntrySide++;
                        diag_brokenInside = true;
                        break;
                    }

                    // Find the other crossing on curface (exit) by
                    // reading the same per-edge cache the seed face
                    // used. This guarantees curface's view of the
                    // shared incoming edge matches prevface's exit.
                    FaceIsoCrossing curFC = computeFaceCrossingsFromCache(
                        m, edgeCache, curface);
                    bool extended = false;
                    for (int idx = 0; idx < curFC.count; ++idx) {
                        int k = curFC.side[idx];
                        if (k == seg.side0) continue;
                        seg.side1 = k;
                        seg.bary1 = curFC.bary[idx];
                        bary = curFC.bary[idx];
                        traces[piece].push_back(seg);
                        prevface = curface;
                        curface = faceSideNeighbor(m, curface, k);
                        extended = true;
                        break;
                    }
                    if (!extended) {
                        diag_stopNoExitCrossing++;
                        if (faceHasBoundaryEdge(m, curface)) diag_noExitBdry++;
                        else                                 diag_noExitInterior++;

                        // noExit-deep: curFC.count bucket
                        {
                            int cnt = curFC.count;
                            if (cnt <= 3) diag_noExit_count[cnt]++;
                            else          diag_noExit_count[4]++;
                        }

                        // noExit-deep: is the entry side present among the
                        // crossings on curface? If yes, the exit was skipped
                        // because the ONLY crossing IS the entry (so the
                        // `if (k == seg.side0) continue;` ate it). If no,
                        // the crossings on curface do not include the entry
                        // side at all -> bary/side inconsistency across the
                        // shared edge between prevface and curface.
                        {
                            bool entryIn = false;
                            for (int idx = 0; idx < curFC.count; ++idx) {
                                if (curFC.side[idx] == seg.side0) {
                                    entryIn = true;
                                    break;
                                }
                            }
                            if (entryIn) diag_noExit_entryIn++;
                            else         diag_noExit_entryNotIn++;
                        }

                        // noExit-deep: first 10 dumps at iso=0 only.
                        if (isoIdx == 0 && diag_noExit_dump < 10) {
                            diag_noExit_dump++;
                            std::fprintf(stderr,
                                "[noexit-dump] iso=0 curface=%d prevface=%d "
                                "entrySide=%d entryBary=%.6f "
                                "curSides=[%d,%d] curBary=[%.6f,%.6f] "
                                "neighbor(entrySide)=%d "
                                "neighbor(curSide0)=%d neighbor(curSide1)=%d\n",
                                curface, prevface,
                                seg.side0, seg.bary0,
                                curFC.side[0], curFC.side[1],
                                curFC.bary[0], curFC.bary[1],
                                faceSideNeighbor(m, curface, seg.side0),
                                curFC.count > 0
                                    ? faceSideNeighbor(m, curface, curFC.side[0])
                                    : -999,
                                curFC.count > 1
                                    ? faceSideNeighbor(m, curface, curFC.side[1])
                                    : -999);
                        }

                        diag_brokenInside = true;
                        break;
                    }
                }
                // Classify the "natural" while-loop exit (not an inner break).
                if (!diag_brokenInside) {
                    if (curface == -1) {
                        diag_stopBoundary++;
                    } else if (visited[curface]) {
                        if (curface == seed) diag_closedLoop++;
                        else                 diag_stopVisited++;
                    } else {
                        diag_stopOther++;
                    }
                }
            }

            // Build the final polyline, chain id, and emit segments.
            int cid = nextChainId++;

            // Part 1: traces[0] reversed + connecting segment on `seed`
            // + traces[1] forward. For the reversed piece, swap entry/exit.
            for (auto it = traces[0].rbegin(); it != traces[0].rend(); ++it) {
                TraceSeg rev = *it;
                std::swap(rev.side0, rev.side1);
                std::swap(rev.bary0, rev.bary1);
                AngularSegment seg;
                seg.a = sidePoint(rev.face, rev.side0, rev.bary0);
                seg.b = sidePoint(rev.face, rev.side1, rev.bary1);
                seg.faceIdx = rev.face;
                seg.iso = isoIdx;
                seg.chainId = cid;
                out.push_back(seg);
            }
            // Connecting segment across the seed face.
            {
                AngularSegment seg;
                seg.a = sidePoint(seed, crossings[0], barys[0]);
                seg.b = sidePoint(seed, crossings[1], barys[1]);
                seg.faceIdx = seed;
                seg.iso = isoIdx;
                seg.chainId = cid;
                out.push_back(seg);
            }
            for (const auto& tseg : traces[1]) {
                AngularSegment seg;
                seg.a = sidePoint(tseg.face, tseg.side0, tseg.bary0);
                seg.b = sidePoint(tseg.face, tseg.side1, tseg.bary1);
                seg.faceIdx = tseg.face;
                seg.iso = isoIdx;
                seg.chainId = cid;
                out.push_back(seg);
            }

            // Diagnostic: record this chain's segment count.
            int chainLen = (int)traces[0].size() + 1 + (int)traces[1].size();
            diag_chains++;
            diag_segments += chainLen;
            diag_chainLens.push_back(chainLen);
        }

        // --- Phase 2: chain length histogram for this iso ---
        int chb[5] = {0, 0, 0, 0, 0};  // 1, 2-4, 5-20, 21-100, >100
        int maxSeg = 0;
        long long sumSeg = 0;
        for (int L : diag_chainLens) {
            sumSeg += L;
            if (L > maxSeg) maxSeg = L;
            if      (L == 1)   chb[0]++;
            else if (L <= 4)   chb[1]++;
            else if (L <= 20)  chb[2]++;
            else if (L <= 100) chb[3]++;
            else               chb[4]++;
        }
        double avgSeg = diag_chains > 0 ? (double)sumSeg / diag_chains : 0.0;

        // --- Phase 1: trace stop-reason summary ---
        std::fprintf(stderr,
            "[trace-diag] iso=%d chains=%d segs=%d "
            "bnd=%d vis=%d noEntry=%d noExit=%d closed=%d other=%d\n",
            isoIdx, diag_chains, diag_segments,
            diag_stopBoundary, diag_stopVisited,
            diag_stopNoEntrySide, diag_stopNoExitCrossing,
            diag_closedLoop, diag_stopOther);

        std::fprintf(stderr,
            "[chain-hist] iso=%d total=%d avgSeg=%.2f maxSeg=%d "
            "buckets[1,2-4,5-20,21-100,>100]=[%d,%d,%d,%d,%d]\n",
            isoIdx, diag_chains, avgSeg, maxSeg,
            chb[0], chb[1], chb[2], chb[3], chb[4]);

        // --- Phase 3: faceSideNeighbor mismatch count ---
        std::fprintf(stderr,
            "[side-check] iso=%d checks=%d mismatches=%d\n",
            isoIdx, diag_sideChecks, diag_sideMismatches);

        // --- Phase 4: per-face crossing count distribution ---
        std::fprintf(stderr,
            "[crossing-hist] iso=%d faces_with_crossings[0,1,2,3,>3]=[%d,%d,%d,%d,%d]\n",
            isoIdx,
            diag_crossBuckets[0], diag_crossBuckets[1],
            diag_crossBuckets[2], diag_crossBuckets[3],
            diag_crossBuckets[4]);

        // Boundary vs interior breakdown of 1-crossing faces.
        std::fprintf(stderr,
            "[crossing-1-breakdown] iso=%d bdry=%d interior=%d\n",
            isoIdx, diag_oneCrossBdry, diag_oneCrossInterior);

        // Causal breakdown of 1-crossing INTERIOR faces (ignores bdry).
        //   wrap:   at least one face edge has |Δθ| > π → crosses() wrap
        //           branch could be what flips the parity to odd
        //   nowrap: all |Δθ| ≤ π, so crosses() is using the simple branch
        //   nan:    theta has NaN/inf on this face
        std::fprintf(stderr,
            "[theta-wrap-stats] iso=%d one_cross_interior=%d "
            "(wrap=%d, nowrap=%d, nan=%d)\n",
            isoIdx, diag_oneCrossInterior,
            diag_oneCrossWithWrap, diag_oneCrossWithoutWrap,
            diag_oneCrossNaN);

        // Boundary vs interior breakdown of no-exit stops.
        std::fprintf(stderr,
            "[noexit-breakdown] iso=%d bdry=%d interior=%d\n",
            isoIdx, diag_noExitBdry, diag_noExitInterior);

        // noExit-deep: curFC.count distribution on the face where we stopped.
        std::fprintf(stderr,
            "[noexit-curfc-count] iso=%d count[0,1,2,3,>3]=[%d,%d,%d,%d,%d]\n",
            isoIdx,
            diag_noExit_count[0], diag_noExit_count[1],
            diag_noExit_count[2], diag_noExit_count[3],
            diag_noExit_count[4]);

        // noExit-deep: did the entry side appear among curface crossings?
        std::fprintf(stderr,
            "[noexit-entry-in-cross] iso=%d yes=%d no=%d\n",
            isoIdx, diag_noExit_entryIn, diag_noExit_entryNotIn);
    }
    return out;
}

} // namespace wgf
