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
#include <queue>
#include <cstdint>

namespace wgf {

struct CoverBuildResult {
    Mesh mesh;                           // cover mesh (all components)
    std::vector<int>   coverFaceBaseF;   // per cover face: base face index
    std::vector<int>   coverFaceLayer;   // per cover face: layer 0..5
    std::vector<int>   coverVertexBaseV; // per cover vertex: base vertex index
    std::vector<int>   coverVertexLayer; // per cover vertex: layer 0..5
    std::vector<char>  isSingularBaseV;  // 1 if base vertex is a branch pt
    std::vector<int8_t> sigmaPerHE;      // σ ∈ 0..5 from base.he[h].face → twin face
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
    R.sigmaPerHE.assign(base.he.size(), 0);
    for (const auto& e : base.eInt) {
        int h = e.first;
        int t = e.second;
        int fi = base.he[h].face;
        int fj = base.he[t].face;
        double beta = parallelTransport(base, baseFrames, h);
        int s = bestShift(phi[fi], phi[fj], beta);
        R.sigmaPerHE[h] = (int8_t)s;
        R.sigmaPerHE[t] = (int8_t)(((6 - s) % 6 + 6) % 6);
    }

    // 2. Holonomy walk around each base vertex.
    //
    // getVertexHalfEdges equivalent: starting at h0 = v2he[v] (outgoing
    // from v), step via he[prev(h)].twin.next, accumulating σ of
    // prev(h) each time (prev(h) ends at v, its σ shifts us into the
    // next fan face).
    //
    // Offsets at each corner (v, f) are recorded for later use.
    R.isSingularBaseV.assign(nV, 0);
    R.numSingular = 0;
    std::unordered_map<long long, int> cornerOffset;
    cornerOffset.reserve((size_t)6 * nF);

    for (int v = 0; v < nV; ++v) {
        int h0 = base.v2he[v];
        if (h0 < 0) continue;

        int acc = 0;  // offset of face(h0) is 0 by convention
        int h = h0;
        bool isBdry = false;

        // First pass: assign offsets around the fan.
        while (true) {
            int f = base.he[h].face;
            long long key = (long long)v * nF + f;
            cornerOffset[key] = ((acc % 6) + 6) % 6;

            int prev = base.he[h].prev;           // ends at v
            int s = R.sigmaPerHE[prev];            // shift into next fan face
            int twin = base.he[prev].twin;
            if (twin < 0) { isBdry = true; break; }
            acc = (acc + s) % 6;
            int next = base.he[twin].next;
            if (next == h0) break;
            h = next;
        }

        // After the loop, acc is the total holonomy (mod 6).
        int holonomy = ((acc % 6) + 6) % 6;
        if (isBdry || holonomy != 0) {
            R.isSingularBaseV[v] = 1;
            R.numSingular++;
            // erase all offsets we stored for this singular vertex
            int hh = h0;
            while (true) {
                int f = base.he[hh].face;
                cornerOffset.erase((long long)v * nF + f);
                int pp = base.he[hh].prev;
                int tt = base.he[pp].twin;
                if (tt < 0) break;
                int nn = base.he[tt].next;
                if (nn == h0) break;
                hh = nn;
            }
        }
    }

    // 3. Compact cover-vertex indices: 6 per non-singular base vertex.
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
