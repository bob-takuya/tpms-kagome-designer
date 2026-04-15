// permutations.h — Vekhter et al. 2019 3-RoSy field → cover permutations.
//
// Faithful port (no algorithmic substitutes) of three reference files from
// https://github.com/the13fools/weaving-geodesic-foliations :
//
//   1. Weave::splitFromRosy (Weave.cpp:943-1052) + RoSyUtils::repVecToRoSy
//      — starting from a single (6-RoSy) direction field on the base mesh,
//      expand per face into three 2D vectors at 60° spacing, with the
//      cyclic alignment chosen by a BFS over faces so that each face's
//      field-0 vector parallel-transports cleanly onto its neighbor's.
//
//   2. Permutations::reassignOnePermutation (Permutations.cpp:80-142)
//      — for each interior edge, exhaustively search S₃ × {±1}³
//      (6×8 = 48 candidates) for the signed 3×3 permutation matrix that
//      best aligns the two face's three 3-RoSy vectors after parallel
//      transport. Matches the reference byte-for-byte.
//
//   3. Permutations::findSingularVertices (Permutations.cpp:159-298)
//      — walks each base vertex's face fan, accumulates the product of
//      edge permutations, flags each field index j at vertex v as
//      singular if the diagonal of the total permutation at (v, j) is
//      not +1 (topological singularity).
//
//   4. Weave::_augmentPs (Weave.cpp:890-920)
//      — turn each 3×3 signed permutation into a 6×6 0/1 permutation
//      matrix on the 6-sheet cover:
//
//         layers 0..2 = (field 0, +), (field 1, +), (field 2, +)
//         layers 3..5 = (field 0, −), (field 1, −), (field 2, −)
//
//      A +1 at Ps(j,k) glues (j+) to (k+) and (j−) to (k−);
//      a −1 at Ps(j,k) glues (j+) to (k−) and (j−) to (k+).
//
// The reference uses Weave / FieldSurface / Surface classes. We use only
// Mesh + FaceFrame and store the per-face 3-RoSy vectors as a flat
// 2*3*|F| = 6|F| vector `wRep3` laid out as
//
//     wRep3[6*f + 2*field + 0] = x component of v(f, field) in face f's 2D frame
//     wRep3[6*f + 2*field + 1] = y component
//
// matching the reference's FieldSurface::vidx(face,field) = 2*m*face + 2*field
// convention (with m = 3).

#pragma once

#include "mesh.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <queue>
#include <array>
#include <cstdio>
#include <algorithm>
#include <cmath>

namespace wgf {

// Index of per-face 3-RoSy vector (field ∈ {0,1,2}) into the flat storage.
inline int repIdx(int face, int field) {
    return 6 * face + 2 * field;   // 2 doubles × 3 fields per face
}

// Read v(f, j) — the j-th 3-RoSy vector on face f in that face's 2D frame.
inline Eigen::Vector2d repVec(const Vec& wRep3, int f, int j) {
    return Eigen::Vector2d(wRep3[repIdx(f,j)], wRep3[repIdx(f,j) + 1]);
}

// -----------------------------------------------------------------------
// 1. splitFromRoSy: single direction field (angle θ_f per face) → three
// per-face vectors at 60° spacing, with BFS-consistent cyclic alignment.
//
// Reference: Weave::splitFromRosy(rosyN=6) + RoSyUtils::repVecToRoSy.
// Behaviour:
//   - For the seed face of each connected component, set v(f, j) to the
//     j-th of six candidate vectors {e^{i(θ + k·2π/6)} : k=0..5} for
//     k = 0, 1, 2.
//   - For every other face, pick the cyclic shift k* ∈ {0..5} such that
//     rv[k*] best matches the parallel-transported predecessor-face's
//     field-0 vector. Then v(f, j) = rv[(j + k*) mod 6] for j = 0, 1, 2.
// -----------------------------------------------------------------------
inline Vec splitFromRoSy(const Mesh& m,
                         const std::vector<FaceFrame>& frames,
                         const Vec& wSingle /* 2|F| unit vectors */) {
    const int F = m.nF();
    const int rosyN = 6;
    const int mFields = 3;
    Vec wRep3 = Vec::Zero(6 * F);

    // Candidate 6-RoSy vectors for face f at angle θ_f = atan2(wSingle[f]):
    //   rv[k] = (cos(θ + k·2π/6), sin(θ + k·2π/6)),  k = 0..5
    auto candidates = [&](int f, std::array<Eigen::Vector2d, 6>& rv) {
        double theta = std::atan2(wSingle[2*f + 1], wSingle[2*f]);
        for (int k = 0; k < rosyN; ++k) {
            double a = theta + 2.0 * M_PI * double(k) / double(rosyN);
            rv[k] = Eigen::Vector2d(std::cos(a), std::sin(a));
        }
    };

    // Helper: 2×2 rotation that takes a 2D vector in fa's frame and
    // produces the same vector in fb's frame via discrete Levi-Civita
    // parallel transport (hinge unfolding around the shared edge).
    // `hab` is a half-edge in fa pointing into the shared edge (i.e.
    // m.he[hab].face == fa and m.he[hab].twin is in fb).
    auto transportFaToFb = [&](int hab) -> Eigen::Matrix2d {
        int twin = m.he[hab].twin;
        int fa = m.he[hab].face;
        int fb = m.he[twin].face;
        Eigen::Vector3d ev =
            m.V.row(m.he[hab].vertex) - m.V.row(m.heStart(hab));
        double aa = std::atan2(ev.dot(frames[fa].e2), ev.dot(frames[fa].e1));
        double ab = std::atan2(ev.dot(frames[fb].e2), ev.dot(frames[fb].e1));
        // angle in fb = angle in fa + (ab - aa) for parallel-transported vectors.
        double beta = ab - aa;
        double c = std::cos(beta), s = std::sin(beta);
        Eigen::Matrix2d T;
        T << c, -s,
             s,  c;
        return T;
    };

    // BFS over faces via twin half-edges. Reference uses breadth-first
    // over face neighbors (Weave.cpp:957-1047).
    std::vector<char> visited(F, 0);
    for (int seedF = 0; seedF < F; ++seedF) {
        if (visited[seedF]) continue;
        struct Visit { int from; int to; int heFromInTo; };
        std::queue<Visit> q;
        q.push({-1, seedF, -1});
        while (!q.empty()) {
            Visit vis = q.front(); q.pop();
            if (visited[vis.to]) continue;
            visited[vis.to] = 1;

            std::array<Eigen::Vector2d, 6> rv;
            candidates(vis.to, rv);

            int bestShift = 0;
            if (vis.from != -1) {
                // Transport vis.from's field-0 vector into vis.to's 2D
                // frame via Levi-Civita hinge unfolding across the
                // shared edge, then find which of vis.to's 6 rosy
                // candidates best matches.
                Eigen::Vector2d v0 = repVec(wRep3, vis.from, 0);

                // heFromInTo is a half-edge in vis.to whose twin is in
                // vis.from. To go from vis.from → vis.to we use its twin.
                int hFrom = m.he[vis.heFromInTo].twin;  // half-edge in vis.from
                Eigen::Matrix2d T = transportFaToFb(hFrom);
                Eigen::Vector2d xportv0 = T * v0;

                // Best cyclic shift: argmax_k rv[k] · xportv0.
                double bestDot = -1e30;
                for (int k = 0; k < rosyN; ++k) {
                    double d = rv[k].dot(xportv0);
                    if (d > bestDot) {
                        bestDot = d;
                        bestShift = k;
                    }
                }
            }

            // Assign fields 0..2 on vis.to.
            for (int j = 0; j < mFields; ++j) {
                int k = (j + bestShift) % rosyN;
                wRep3[repIdx(vis.to, j)]     = rv[k].x();
                wRep3[repIdx(vis.to, j) + 1] = rv[k].y();
            }

            // Queue unvisited neighbors. For each neighbor nb, record
            // the half-edge in nb whose twin is in vis.to (used by the
            // transport step when nb is later dequeued).
            int h0 = m.f2he[vis.to];
            for (int i = 0; i < 3; ++i) {
                int heInTo = h0 + i;
                int twin = m.he[heInTo].twin;
                if (twin < 0) continue;
                int nb = m.he[twin].face;
                if (!visited[nb]) {
                    // From nb's perspective, the shared half-edge is
                    // `twin` (lives in nb).
                    q.push({vis.to, nb, twin});
                }
            }
        }
    }

    return wRep3;
}

// -----------------------------------------------------------------------
// 2. reassignOnePermutation: for a given interior edge with half-edges
// (h, twin(h)), with fi = face(h), fj = face(twin(h)), find the 3×3
// signed permutation P ∈ {-1, 0, +1}^{3×3} (exactly one non-zero per
// row and per column) that minimises
//
//     Σ_{j=0..2}  angle²(v(fi, j),  P(j,·) · T · v(fj, ·))
//
// where T is the rotation that brings vectors from fj's 2D frame into
// fi's 2D frame (parallel transport), and angle(a,b) is the signed
// planar angle from a to b in fi's 2D frame.
//
// Searches all 3! × 2³ = 48 (permutation, sign-pattern) pairs.
// Matches Permutations::reassignOnePermutation line-for-line.
// -----------------------------------------------------------------------

// Signed angle from a to b in fi's 2D plane, in (−π, π].
// Mirrors RoSyUtils::angle: 2·atan2(cross, dot + |a||b|), where cross
// is the 2D cross-product (z-component of 3D cross).
inline double signedAngle2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    double cross = a.x() * b.y() - a.y() * b.x();
    double dot   = a.dot(b);
    double normP = a.norm() * b.norm();
    return 2.0 * std::atan2(cross, normP + dot);
}

inline Eigen::Matrix3i reassignOnePermutation(
    const Mesh& m,
    const std::vector<FaceFrame>& frames,
    const Vec& wRep3,
    int h /* half-edge */) {

    Eigen::Matrix3i P = Eigen::Matrix3i::Zero();
    int t = m.he[h].twin;
    if (t < 0) {
        P.setIdentity();
        return P;
    }
    int fi = m.he[h].face;
    int fj = m.he[t].face;

    // Transport rotation: given a 2D vector u_g in fj's frame, produce
    // the same 3D vector expressed in fi's 2D frame. Rotation is
    // R(ai − aj) where ai, aj are the angles of the shared edge in each
    // face's frame. In matrix form: [[c, -s], [s, c]] with
    // c = cos(ai−aj), s = sin(ai−aj).
    Eigen::Vector3d ev = m.V.row(m.he[h].vertex) - m.V.row(m.heStart(h));
    double ai = std::atan2(ev.dot(frames[fi].e2), ev.dot(frames[fi].e1));
    double aj = std::atan2(ev.dot(frames[fj].e2), ev.dot(frames[fj].e1));
    double cb = std::cos(ai - aj);
    double sb = std::sin(ai - aj);
    Eigen::Matrix2d T;
    T << cb, -sb,
         sb,  cb;

    // Gather vectors: fvecs[i] = v(fi, i) in fi's frame (raw 2D).
    //                gvecs[i] = T * v(fj, i) = v(fj, i) transported into fi's frame.
    std::array<Eigen::Vector2d, 3> fvecs, gvecs;
    for (int i = 0; i < 3; ++i) {
        fvecs[i] = repVec(wRep3, fi, i);
        gvecs[i] = T * repVec(wRep3, fj, i);
    }

    // Exhaustive search over S_3 × {±1}^3 = 48 candidates.
    double bestCost = std::numeric_limits<double>::infinity();
    int bestSigns = 0;
    std::array<int, 3> bestPerm{0, 1, 2};

    std::array<int, 3> perm{0, 1, 2};
    do {
        for (int signs = 0; signs < 8; ++signs) {
            double cost = 0.0;
            for (int i = 0; i < 3; ++i) {
                double sign = (signs & (1 << i)) ? -1.0 : 1.0;
                double theta = signedAngle2d(fvecs[i], sign * gvecs[perm[i]]);
                cost += theta * theta;
            }
            if (cost < bestCost) {
                bestCost  = cost;
                bestSigns = signs;
                bestPerm  = perm;
            }
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    for (int i = 0; i < 3; ++i) {
        int sign = (bestSigns & (1 << i)) ? -1 : 1;
        P(i, bestPerm[i]) = sign;
    }
    return P;
}

// Compute one 3×3 signed permutation per interior edge, indexed by
// interior-edge position in m.eInt.  Returns a vector of size m.nE().
inline std::vector<Eigen::Matrix3i>
reassignAllPermutations(const Mesh& m,
                        const std::vector<FaceFrame>& frames,
                        const Vec& wRep3) {
    const int nE = m.nE();
    std::vector<Eigen::Matrix3i> Ps(nE);
    for (int e = 0; e < nE; ++e) {
        int h = m.eInt[e].first;
        Ps[e] = reassignOnePermutation(m, frames, wRep3, h);
    }
    return Ps;
}

// Diagnostic variant: also records the minimum cost (residual angle²)
// at each edge, so callers can see how cleanly the 3-RoSy field aligns
// across neighbor faces.  Used by smoke_native for investigation.
inline std::vector<Eigen::Matrix3i>
reassignAllPermutationsDebug(const Mesh& m,
                             const std::vector<FaceFrame>& frames,
                             const Vec& wRep3,
                             Vec& bestCostsOut) {
    const int nE = m.nE();
    std::vector<Eigen::Matrix3i> Ps(nE);
    bestCostsOut.resize(nE);
    bestCostsOut.setZero();
    for (int e = 0; e < nE; ++e) {
        int h = m.eInt[e].first;
        int t = m.he[h].twin;
        if (t < 0) { Ps[e].setIdentity(); continue; }
        int fi = m.he[h].face;
        int fj = m.he[t].face;
        Eigen::Vector3d ev = m.V.row(m.he[h].vertex) - m.V.row(m.heStart(h));
        double ai = std::atan2(ev.dot(frames[fi].e2), ev.dot(frames[fi].e1));
        double aj = std::atan2(ev.dot(frames[fj].e2), ev.dot(frames[fj].e1));
        double cb = std::cos(ai - aj), sb = std::sin(ai - aj);
        Eigen::Matrix2d T;
        T << cb, -sb, sb, cb;
        std::array<Eigen::Vector2d, 3> fvecs, gvecs;
        for (int i = 0; i < 3; ++i) {
            fvecs[i] = repVec(wRep3, fi, i);
            gvecs[i] = T * repVec(wRep3, fj, i);
        }
        double bestCost = std::numeric_limits<double>::infinity();
        int bestSigns = 0;
        std::array<int, 3> bestPerm{0, 1, 2};
        std::array<int, 3> perm{0, 1, 2};
        do {
            for (int signs = 0; signs < 8; ++signs) {
                double cost = 0.0;
                for (int i = 0; i < 3; ++i) {
                    double sign = (signs & (1 << i)) ? -1.0 : 1.0;
                    double theta = signedAngle2d(fvecs[i], sign * gvecs[perm[i]]);
                    cost += theta * theta;
                }
                if (cost < bestCost) {
                    bestCost = cost;
                    bestSigns = signs;
                    bestPerm = perm;
                }
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
        Ps[e].setZero();
        for (int i = 0; i < 3; ++i) {
            int sign = (bestSigns & (1 << i)) ? -1 : 1;
            Ps[e](i, bestPerm[i]) = sign;
        }
        bestCostsOut[e] = bestCost;
    }
    return Ps;
}

// -----------------------------------------------------------------------
// 3. findSingularVertices — per-field topological singularity detection.
//
// For each non-boundary vertex v, walk its 1-ring of incident faces,
// accumulating the product of per-edge permutations. If the total
// permutation is not the identity, (v, field) with totperm(j,j) != 1
// is flagged singular.
//
// Returns, per vertex, a bitmask of singular field indices in bits 0..2
// (so 0 = non-singular at all 3 fields). The caller turns this into
// punctured cover layers via augmentPs's block structure.
//
// Reference: Permutations::findSingularVertices (topological half only).
// -----------------------------------------------------------------------
inline std::vector<int>
findSingularFieldsPerVertex(
    const Mesh& m,
    const std::vector<Eigen::Matrix3i>& Ps /* per interior edge, length m.nE() */) {

    const int nV = m.nV();
    std::vector<int> out(nV, 0);

    // Map each half-edge to its interior-edge index and "side":
    //   side = 0 if E(edge, 0) = face(h), side = 1 if E(edge, 1) = face(h).
    // E(edge, 0) is defined as face(m.eInt[edge].first).
    std::vector<int>  heToEdge((int)m.he.size(), -1);
    std::vector<char> heSide((int)m.he.size(), 0);
    for (int e = 0; e < (int)m.eInt.size(); ++e) {
        int h0 = m.eInt[e].first;
        int h1 = m.eInt[e].second;
        heToEdge[h0] = e;  heSide[h0] = 0;
        heToEdge[h1] = e;  heSide[h1] = 1;
    }

    const int walkGuard = (int)m.he.size() + 16;

    for (int v = 0; v < nV; ++v) {
        int h0 = m.v2he[v];
        if (h0 < 0) continue;

        Eigen::Matrix3i totPerm = Eigen::Matrix3i::Identity();
        bool isBdry = false;
        int steps = 0;

        int h = h0;
        while (true) {
            if (++steps > walkGuard) { isBdry = true; break; }
            // Cross the shared edge on the "prev" side (edge separating
            // face(h) from the next face in the fan, see cover.h for the
            // fan-walk geometry). Reference uses faceEdges(curface,
            // curspoke) and the E(edge, 0)==curface test.
            int prev = m.he[h].prev;              // ends at v
            int twin = m.he[prev].twin;           // next outgoing half-edge from v
            if (twin < 0) { isBdry = true; break; }
            int edge = heToEdge[prev];
            if (edge < 0) { isBdry = true; break; }
            // If prev is on side 0, fs->Ps(e) is already the (curface → nextface)
            // mapping in the reference sense (see Permutations.cpp:217-224:
            //   side == 0 : nextperm = Ps.transpose()
            //   side == 1 : nextperm = Ps
            // because Ps maps E(e,1) → E(e,0), so when curface = E(e,0)
            // we need the inverse = transpose).
            Eigen::Matrix3i nextPerm;
            if (heSide[prev] == 0)
                nextPerm = Ps[edge].transpose();
            else
                nextPerm = Ps[edge];

            totPerm = totPerm * nextPerm;

            if (twin == h0) break;
            h = twin;
        }

        if (isBdry) {
            // Boundary vertices: mark all three fields as singular so we
            // puncture them just like the reference's _augmentPs flow.
            out[v] = 0b111;
            continue;
        }
        int mask = 0;
        for (int j = 0; j < 3; ++j) {
            if (totPerm(j, j) != 1) mask |= (1 << j);
        }
        out[v] = mask;
    }
    return out;
}

// -----------------------------------------------------------------------
// 4. augmentPs: 3×3 signed permutation → 6×6 0/1 cover permutation.
//
// Reference: Weave::_augmentPs (Weave.cpp:890-920). Block layout:
//
//   perm(j,  k)   = 1   if  Ps(j,k) == +1
//   perm(j+3,k+3) = 1   if  Ps(j,k) == +1
//   perm(j,  k+3) = 1   if  Ps(j,k) == -1
//   perm(j+3,k)   = 1   if  Ps(j,k) == -1
//
// i.e. layers 0..2 are the +1 copies of fields 0..2, layers 3..5 are
// the −1 copies, and a sign flip on Ps(j,k) swaps between the two
// halves of the cover.
// -----------------------------------------------------------------------
using CoverPerm = Eigen::Matrix<int, 6, 6>;

inline CoverPerm augmentOnePs(const Eigen::Matrix3i& P3) {
    CoverPerm P6 = CoverPerm::Zero();
    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
            int s = P3(j, k);
            if (s == 1) {
                P6(j,     k    ) = 1;
                P6(j + 3, k + 3) = 1;
            } else if (s == -1) {
                P6(j,     k + 3) = 1;
                P6(j + 3, k    ) = 1;
            }
        }
    }
    return P6;
}

inline std::vector<CoverPerm>
augmentPs(const std::vector<Eigen::Matrix3i>& Ps) {
    std::vector<CoverPerm> out;
    out.reserve(Ps.size());
    for (const auto& P3 : Ps) out.push_back(augmentOnePs(P3));
    return out;
}

// -----------------------------------------------------------------------
// Diagnostic: histogram the per-edge 6×6 cover permutations into three
// buckets for logging.  Matches the cover.h reporting format.
//
//   identity   : P = I                   (neither field nor sign change)
//   sign_only  : permutes within a field (Ps = ±I, not I)
//   permuting  : non-trivial field perm  (not a sign-only matrix)
// -----------------------------------------------------------------------
struct PsHistogram {
    int identity     = 0;
    int signOnly     = 0;  // Ps = diag(±1, ±1, ±1), not all +1
    int permuting    = 0;
};

inline PsHistogram histogramPs(const std::vector<Eigen::Matrix3i>& Ps) {
    PsHistogram H;
    Eigen::Matrix3i I = Eigen::Matrix3i::Identity();
    for (const auto& P : Ps) {
        if (P == I) { H.identity++; continue; }
        bool diag = true;
        for (int i = 0; i < 3 && diag; ++i) {
            for (int j = 0; j < 3 && diag; ++j) {
                if (i == j) {
                    if (P(i,i) != 1 && P(i,i) != -1) diag = false;
                } else {
                    if (P(i,j) != 0) diag = false;
                }
            }
        }
        if (diag) H.signOnly++;
        else      H.permuting++;
    }
    return H;
}

} // namespace wgf
