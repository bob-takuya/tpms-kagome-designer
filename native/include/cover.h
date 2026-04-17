// cover.h — 6-fold branched cover construction (Vekhter 2019 §5.1).
//
// For a 6-RoSy direction field with a canonical angle φ_f per face, the
// 6 layer directions are rigid 60° rotations of each other. The
// permutation σ_{ij} ∈ S₆ that best aligns the two sets of layers
// across an interior edge therefore reduces to a cyclic shift in ℤ/6ℤ
// (no reflection component, because reversing the line direction on a
// 3-line kagome field is already the rotation r³).
//
// For each interior edge with parallel-transport rotation β_{ij}:
//
//     σ_{ij} = round( (φ_i − φ_j + β_{ij}) / (π/3) )   (mod 6)
//
// A base vertex v is a BRANCH POINT iff the total holonomy around v
// (the sum of σ's along the incident half-edge fan) is non-zero mod 6.
//
// Branch-point vertices and every face incident to them are punctured
// from the cover. For every surviving face we emit 6 layer copies glued
// together via σ. The resulting half-edge mesh can be fed back into
// solve6RoSy / buildCurlOperator / runAlg1 without modification.

#pragma once

#include "mesh.h"
#include "rosy.h"
#include "progress.h"
#include <queue>
#include <cstdint>
#include <cstdio>

namespace wgf {

struct CoverBuildResult {
    Mesh mesh;                           // cover mesh (all components)
    std::vector<int>   coverFaceBaseF;   // per cover face: base face index
    std::vector<int>   coverFaceLayer;   // per cover face: layer 0..5
    std::vector<int>   coverVertexBaseV; // per cover vertex: base vertex index
    std::vector<int>   coverVertexLayer; // per cover vertex: layer 0..5
    std::vector<char>  isSingularBaseV;  // 1 if base vertex is a branch pt
    std::vector<int8_t> sigmaPerHE;      // σ ∈ 0..5 from base.he[h].face → twin face
    // Direct lookup: flatBaseIdx[6*v_base + k] = cover vertex index, or -1
    // if base vertex v_base is singular. Used by paired_rounding.h to find
    // antipodal layer pairs (v, k) ↔ (v, k+3).
    std::vector<int>   flatBaseIdx;
    int numSingular;
};

// Cyclic shift σ_ij ∈ ℤ/6ℤ best aligning the two 6-RoSy layer sets.
// Returns an integer in [0, 5].
inline int bestShift(double phi_i, double phi_j, double beta) {
    const double SIXTH = M_PI / 3.0;
    double rawShift = (phi_i + beta - phi_j) / SIXTH;
    int s = (int)std::llround(rawShift);
    return ((s % 6) + 6) % 6;
}

inline CoverBuildResult buildBranchedCover(
    const Mesh& base,
    const std::vector<FaceFrame>& baseFrames,
    const Vec& phi /* per-face 6-RoSy angle */) {

    CoverBuildResult R;
    const int nV = base.nV();
    const int nF = base.nF();

    // 1. σ per directed half-edge.
    //
    // σ_h = shift from face(h) to face(twin(h)).  We only touch each
    // interior edge once and set both h and twin(h) consistently:
    // σ_{twin(h)} = (6 − σ_h) mod 6.
    reportProgress(STAGE_COVER_BUILD, 0, 4, "Cover: edge permutations");
    R.sigmaPerHE.assign(base.he.size(), 0);
    int sigmaHist[6] = {0, 0, 0, 0, 0, 0};
    double residualMaxRad = 0;
    double residualSumSq  = 0;
    for (const auto& e : base.eInt) {
        int h = e.first;
        int t = e.second;
        int fi = base.he[h].face;
        int fj = base.he[t].face;
        double beta = parallelTransport(base, baseFrames, h);
        // Raw angular difference (before rounding). Good diagnostic —
        // if |raw − k·π/3| is consistently < π/6 at every edge then
        // the cover build will be clean.
        const double SIXTH = M_PI / 3.0;
        double raw    = phi[fi] + beta - phi[fj];
        int    sRound = (int)std::llround(raw / SIXTH);
        double resid  = raw - sRound * SIXTH;
        // Wrap residual into (−π/6, π/6].
        while (resid >  M_PI / 6.0 + 1e-12) resid -= SIXTH;
        while (resid < -M_PI / 6.0 - 1e-12) resid += SIXTH;
        if (std::fabs(resid) > residualMaxRad) residualMaxRad = std::fabs(resid);
        residualSumSq += resid * resid;

        int s = ((sRound % 6) + 6) % 6;
        R.sigmaPerHE[h] = (int8_t)s;
        R.sigmaPerHE[t] = (int8_t)(((6 - s) % 6 + 6) % 6);
        sigmaHist[s]++;
    }
    {
        int nE = (int)base.eInt.size();
        double rms = std::sqrt(residualSumSq / std::max(1, nE));
        std::fprintf(stderr,
            "[cover] sigma hist [%d %d %d %d %d %d] / %d edges | residual max=%.3g rad (%.1f deg) rms=%.3g\n",
            sigmaHist[0], sigmaHist[1], sigmaHist[2],
            sigmaHist[3], sigmaHist[4], sigmaHist[5],
            nE, residualMaxRad, residualMaxRad * 180.0 / M_PI, rms);
    }

    // 2. Holonomy walk around each base vertex.
    //
    // Given an outgoing half-edge h (h starts at v, ends at some u),
    // the canonical CW-around-v rotation is
    //
    //     h_next = twin(prev(h))
    //
    // prev(h) is the half-edge ending at v in face(h); its twin starts
    // at v in the adjacent face. That adjacent-face half-edge IS the
    // next outgoing half-edge from v, so we just walk `h ← twin(prev(h))`
    // until we cycle back to h0.
    //
    // An earlier version used `h ← next(twin(prev(h)))` which drifts
    // off the fan of v entirely (next(twin) starts at twin's endpoint,
    // not at v), causing the walk to wander the whole mesh for each
    // vertex — O(|V|·|F|) in the worst case, effectively hanging on
    // real TPMS meshes.
    reportProgress(STAGE_COVER_BUILD, 1, 4, "Cover: vertex holonomy walk");
    R.isSingularBaseV.assign(nV, 0);
    R.numSingular = 0;
    std::unordered_map<long long, int> cornerOffset;
    cornerOffset.reserve((size_t)3 * nF);   // ~3 corner entries per face

    const int walkGuard = (int)base.he.size() + 16;  // defensive cap
    const int reportEveryV = std::max(1, nV / 20);
    int holHist[6] = {0, 0, 0, 0, 0, 0};
    int nBdry = 0;

    for (int v = 0; v < nV; ++v) {
        if ((v % reportEveryV) == 0) {
            reportProgress(STAGE_COVER_BUILD, 1, 4, "Cover: vertex holonomy walk");
        }
        int h0 = base.v2he[v];
        if (h0 < 0) continue;

        // For a boundary vertex, v2he[v] may land anywhere in the open fan.
        // The forward walk `h ← twin(prev(h))` terminates at the CW-end
        // (where prev(h) is a boundary edge) but never crosses the CCW-end,
        // so starting in the middle leaves the CCW-side of the fan with no
        // cornerOffset entry — and those faces silently vanish from the
        // cover. Rewind via the inverse rotation `h ← next(twin(h))` until
        // we hit a boundary (CCW-end) or cycle back (closed fan).
        {
            int hRew = h0;
            for (int g = 0; g < walkGuard; ++g) {
                int tw = base.he[hRew].twin;
                if (tw < 0) break;                  // reached CCW-end of fan
                int nxt = base.he[tw].next;
                if (nxt == h0) break;               // closed fan, original h0 fine
                hRew = nxt;
            }
            h0 = hRew;
        }

        int acc = 0;  // offset of face(h0) is 0 by convention
        int h = h0;
        bool isBdry = false;
        int steps = 0;

        // First pass: assign offsets around the fan.
        while (true) {
            if (++steps > walkGuard) { isBdry = true; break; }  // should never fire
            int f = base.he[h].face;
            long long key = (long long)v * nF + f;
            cornerOffset[key] = ((acc % 6) + 6) % 6;

            int prev = base.he[h].prev;            // ends at v
            int twin = base.he[prev].twin;         // next outgoing from v
            if (twin < 0) { isBdry = true; break; }
            int s = R.sigmaPerHE[prev];            // σ across the shared edge
            acc = (acc + s) % 6;
            if (twin == h0) break;                 // closed the fan
            h = twin;
        }

        // After the loop, acc is the total holonomy (mod 6).
        int holonomy = ((acc % 6) + 6) % 6;
        holHist[holonomy]++;
        if (isBdry) nBdry++;
        // Only true topological singularities (interior vertices with non-zero
        // holonomy) become branch points. Boundary vertices have an open fan
        // and θ terminates naturally at the boundary, so puncturing them just
        // shatters the cover into many components.
        if (!isBdry && holonomy != 0) {
            R.isSingularBaseV[v] = 1;
            R.numSingular++;
            // erase all offsets we stored for this singular vertex
            int hh = h0;
            int g2 = 0;
            while (true) {
                if (++g2 > walkGuard) break;
                int f = base.he[hh].face;
                cornerOffset.erase((long long)v * nF + f);
                int pp = base.he[hh].prev;
                int tt = base.he[pp].twin;
                if (tt < 0) break;
                if (tt == h0) break;
                hh = tt;
            }
        }
    }

    std::fprintf(stderr,
        "[cover] holonomy hist [%d %d %d %d %d %d] / %d vertices (boundary=%d) singular=%d\n",
        holHist[0], holHist[1], holHist[2], holHist[3], holHist[4], holHist[5],
        nV, nBdry, R.numSingular);

    // 3. Compact cover-vertex indices: 6 per non-singular base vertex.
    reportProgress(STAGE_COVER_BUILD, 2, 4, "Cover: compact vertices");
    std::vector<int> flat(6 * nV, -1);
    std::vector<int> cvBase, cvLayer;
    cvBase.reserve(6 * nV);
    cvLayer.reserve(6 * nV);
    for (int v = 0; v < nV; ++v) {
        if (R.isSingularBaseV[v]) continue;
        for (int k = 0; k < 6; ++k) {
            flat[6*v + k] = (int)cvBase.size();
            cvBase.push_back(v);
            cvLayer.push_back(k);
        }
    }
    const int nCV = (int)cvBase.size();

    MatV CV(nCV, 3);
    for (int i = 0; i < nCV; ++i) CV.row(i) = base.V.row(cvBase[i]);

    // 4. Cover faces: 6 copies of every non-punctured face.
    reportProgress(STAGE_COVER_BUILD, 3, 4, "Cover: emit faces + half-edges");
    std::vector<Eigen::Vector3i> CF;
    std::vector<int> fbase, flayer;
    CF.reserve(6 * nF);
    fbase.reserve(6 * nF);
    flayer.reserve(6 * nF);

    for (int f = 0; f < nF; ++f) {
        int v0 = base.F(f, 0), v1 = base.F(f, 1), v2 = base.F(f, 2);
        if (R.isSingularBaseV[v0] ||
            R.isSingularBaseV[v1] ||
            R.isSingularBaseV[v2]) continue;

        auto it0 = cornerOffset.find((long long)v0 * nF + f);
        auto it1 = cornerOffset.find((long long)v1 * nF + f);
        auto it2 = cornerOffset.find((long long)v2 * nF + f);
        if (it0 == cornerOffset.end() ||
            it1 == cornerOffset.end() ||
            it2 == cornerOffset.end()) continue;
        int o0 = it0->second;
        int o1 = it1->second;
        int o2 = it2->second;

        for (int k = 0; k < 6; ++k) {
            int l0 = ((k + o0) % 6 + 6) % 6;
            int l1 = ((k + o1) % 6 + 6) % 6;
            int l2 = ((k + o2) % 6 + 6) % 6;
            int a = flat[6*v0 + l0];
            int b = flat[6*v1 + l1];
            int c = flat[6*v2 + l2];
            if (a < 0 || b < 0 || c < 0) continue;
            CF.emplace_back(a, b, c);
            fbase.push_back(f);
            flayer.push_back(k);
        }
    }

    R.mesh.V = CV;
    R.mesh.F.resize((int)CF.size(), 3);
    for (int i = 0; i < (int)CF.size(); ++i) R.mesh.F.row(i) = CF[i];
    buildHalfEdges(R.mesh);

    R.coverFaceBaseF   = std::move(fbase);
    R.coverFaceLayer   = std::move(flayer);
    R.coverVertexBaseV = std::move(cvBase);
    R.coverVertexLayer = std::move(cvLayer);
    R.flatBaseIdx      = std::move(flat);
    return R;
}

// Label connected components of a mesh by face BFS through twin edges.
inline int labelComponents(const Mesh& m, std::vector<int>& faceComp) {
    const int nF = m.nF();
    faceComp.assign(nF, -1);
    int nComp = 0;
    for (int seed = 0; seed < nF; ++seed) {
        if (faceComp[seed] >= 0) continue;
        std::queue<int> q;
        q.push(seed);
        faceComp[seed] = nComp;
        while (!q.empty()) {
            int f = q.front(); q.pop();
            int h = m.f2he[f];
            for (int k = 0; k < 3; ++k) {
                int twin = m.he[h].twin;
                if (twin >= 0) {
                    int nf = m.he[twin].face;
                    if (faceComp[nf] < 0) {
                        faceComp[nf] = nComp;
                        q.push(nf);
                    }
                }
                h = m.he[h].next;
            }
        }
        nComp++;
    }
    return nComp;
}

// Extract a sub-mesh keeping only faces labeled `target`.
struct SubMesh {
    Mesh mesh;
    std::vector<int> oldFaceIdx;    // new face → old face
    std::vector<int> oldVertexIdx;  // new vertex → old vertex
};

inline SubMesh extractComponent(const Mesh& m,
                                const std::vector<int>& faceComp,
                                int target) {
    std::vector<int> newV(m.nV(), -1);
    std::vector<int> oldV;
    std::vector<Eigen::Vector3i> Fsub;
    std::vector<int> oldF;
    for (int f = 0; f < m.nF(); ++f) {
        if (faceComp[f] != target) continue;
        Eigen::Vector3i tri;
        for (int k = 0; k < 3; ++k) {
            int v = m.F(f, k);
            if (newV[v] < 0) {
                newV[v] = (int)oldV.size();
                oldV.push_back(v);
            }
            tri[k] = newV[v];
        }
        Fsub.push_back(tri);
        oldF.push_back(f);
    }
    SubMesh S;
    S.mesh.V.resize((int)oldV.size(), 3);
    for (int i = 0; i < (int)oldV.size(); ++i) S.mesh.V.row(i) = m.V.row(oldV[i]);
    S.mesh.F.resize((int)Fsub.size(), 3);
    for (int i = 0; i < (int)Fsub.size(); ++i) S.mesh.F.row(i) = Fsub[i];
    buildHalfEdges(S.mesh);
    S.oldFaceIdx   = std::move(oldF);
    S.oldVertexIdx = std::move(oldV);
    return S;
}

} // namespace wgf
