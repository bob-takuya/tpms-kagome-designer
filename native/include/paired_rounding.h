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

    // Gauss-Seidel rounding (Bommes 2009 §4.3): round one batch of
    // integer variables per iteration and re-solve. To keep the total
    // work bounded, we round in batches of ~5% of ncorrs so we make
    // 20 passes total rather than ncorrs.
    std::vector<char>   fixedMask(ncorrs, 0);
    std::vector<double> nFixed(ncorrs, 0.0);
    int numRounded = 0;
    const int batchSize = std::max(1, ncorrs / 20);
    int passes = 0;

    Vec delta, nCont;
    while (numRounded < ncorrs) {
        ++passes;
        reportProgress(STAGE_ASSEMBLE, numRounded, ncorrs,
                       "Paired rounding (Gauss-Seidel)");
        if (!solveContinuousRelaxation(L, LepsFactor, pairs, fixedMask, nFixed,
                                       theta, phase, offset, delta, nCont)) {
            std::fprintf(stderr, "[paired-rounding] CG failed at pass %d\n", passes);
            break;
        }

        // Collect rounding errors for non-fixed pairs and pick the
        // `batchSize` smallest ones.
        std::vector<std::pair<double,int>> errs;   // (|err|, pairIndex)
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

    std::fprintf(stderr, "[paired-rounding] finished in %d passes\n", passes);
}

// Extract isolines from a CIRCULAR (periodic) angular scalar field.
//
// For a real-valued theta in (-pi, pi], level sets at
//     val = -pi + k * (2pi / numISOLines),  k = 0 .. numISOLines - 1
// are extracted. Each triangle edge whose two endpoints straddle the
// target val (accounting for the 2pi wrap) contributes one crossing,
// and triangles with exactly two crossings emit one segment.
//
// The "straddle" predicate handles the periodic case: if the two endpoint
// values differ by more than pi, we unwrap one of them by adding 2pi
// and re-test. This matches the reference CoverMesh::crosses().
struct AngularSegment {
    Eigen::Vector3d a, b;
    int faceIdx;
    int iso;  // which level index 0..numISOLines-1
};

inline std::vector<AngularSegment>
extractAngularIsolines(const Mesh& m, const Vec& theta, int numISOLines) {
    std::vector<AngularSegment> out;
    const int nF = m.nF();
    const double minval = -M_PI;
    const double maxval =  M_PI;

    auto crosses = [&](double isoval, double v1, double v2, double& bary) -> bool {
        const double halfperiod = M_PI;
        if (std::fabs(v2 - v1) <= halfperiod) {
            bary = (isoval - v1) / (v2 - v1);
            return bary >= 0.0 && bary < 1.0;
        }
        if (v1 < v2) {
            double w1 = v1 + 2 * M_PI;
            bary = (isoval - w1) / (v2 - w1);
            if (bary >= 0.0 && bary < 1.0) return true;
            double w2 = v2 - 2 * M_PI;
            bary = (isoval - v1) / (w2 - v1);
            if (bary >= 0.0 && bary < 1.0) return true;
        } else {
            double w1 = v1 - 2 * M_PI;
            bary = (isoval - w1) / (v2 - w1);
            if (bary >= 0.0 && bary < 1.0) return true;
            double w2 = v2 + 2 * M_PI;
            bary = (isoval - v1) / (w2 - v1);
            if (bary >= 0.0 && bary < 1.0) return true;
        }
        return false;
    };

    for (int k = 0; k < numISOLines; ++k) {
        const double isoval = minval + (maxval - minval) * double(k) / double(numISOLines);

        for (int f = 0; f < nF; ++f) {
            int v[3] = { m.F(f, 0), m.F(f, 1), m.F(f, 2) };
            double r[3] = { theta[v[0]], theta[v[1]], theta[v[2]] };
            Eigen::Vector3d p[3] = {
                m.V.row(v[0]), m.V.row(v[1]), m.V.row(v[2])
            };

            Eigen::Vector3d cp[3];
            int nCross = 0;
            for (int e = 0; e < 3; ++e) {
                int a = e, b = (e + 1) % 3;
                double bary;
                if (crosses(isoval, r[a], r[b], bary)) {
                    if (nCross < 3)
                        cp[nCross] = p[a] + bary * (p[b] - p[a]);
                    nCross++;
                }
            }
            if (nCross == 2) {
                AngularSegment s;
                s.a = cp[0];
                s.b = cp[1];
                s.faceIdx = f;
                s.iso = k;
                out.push_back(s);
            }
        }
    }
    return out;
}

} // namespace wgf
