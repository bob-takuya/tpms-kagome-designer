// cover.h — 6-fold branched cover construction (Vekhter 2019 §5.1).
//
// Faithful port of Weave::createCover (Weave.cpp:687-857) from the
// reference C++ code at
//     https://github.com/the13fools/weaving-geodesic-foliations
//
// Input:
//   - A base triangle mesh + per-face frames.
//   - A per-edge 6×6 0/1 permutation matrix `Pcover[e]` for each interior
//     edge (one per entry in m.eInt). These are produced by
//     permutations.h::augmentPs from a set of per-edge 3×3 signed
//     permutations (Permutations::reassignOnePermutation).
//   - A per-vertex mask `singularFields[v]` with bit j set if field j
//     at vertex v is topologically singular (cover layers j and j+3
//     will be punctured around v).
//
// Output: a CoverBuildResult that contains
//   - the flattened 6-sheet cover mesh (post-gluing),
//   - per-cover-face and per-cover-vertex back-refs into the base mesh,
//   - a compact flatBaseIdx[] lookup for paired_rounding.h to find each
//     non-singular base vertex's 6 cover copies.
//
// Cover construction proceeds in two phases, matching the reference:
//
//   A. Gluing. For every interior edge e, the 6×6 `Pcover[e]` tells us
//      which layer on the E(e,0)-side face should be glued to which
//      layer on the E(e,1)-side face. We build an adjacency list over
//      3·|F|·6 "corner-layer" nodes (one per (face, local corner,
//      layer)) — identical to the reference's `adj_list` in
//      Weave.cpp:697-747 — and BFS it to extract connected components.
//      Each connected component becomes a single cover vertex.
//
//   B. Emission. For every non-punctured cover face, we look up its
//      three corners' merged cover-vertex indices and emit a triangle.
//      Punctured faces are those with any singular corner layer at any
//      of their 3 incident vertices.
//
// For paired rounding we also compute, per non-singular base vertex v,
// which cover vertex carries each layer k ∈ {0..5}. This is simply the
// connected component containing the (v, incident-face, corner, k)
// node for any face at v — by construction of Pcover (and the trivial
// monodromy at non-singular vertices), this is independent of the
// chosen incident face.

#pragma once

#include "mesh.h"
#include "permutations.h"
#include "progress.h"
#include <queue>
#include <cstdint>
#include <cstdio>
#include <unordered_map>

namespace wgf {

struct CoverBuildResult {
    Mesh mesh;                           // cover mesh (all components)
    std::vector<int>   coverFaceBaseF;   // per cover face: base face index
    std::vector<int>   coverFaceLayer;   // per cover face: layer 0..5
    std::vector<int>   coverVertexBaseV; // per cover vertex: base vertex index
    std::vector<int>   coverVertexLayer; // per cover vertex: layer 0..5
    std::vector<char>  isSingularBaseV;  // 1 if base vertex has any singular field
    // flatBaseIdx[6*v + k] = cover vertex index for non-singular (v, k),
    // or -1 if base vertex v has field (k mod 3) marked singular, or the
    // base vertex is on a boundary.  Used by paired_rounding.h.
    std::vector<int>   flatBaseIdx;
    int numSingular;
};

// -----------------------------------------------------------------------
// Encode / decode (face, corner, layer) into a flat node index for the
// BFS adjacency list. Layout: layer-major, face-inner, corner-innermost
// (same as the reference):
//
//   encodedId = corner + 3 * face + 3 * nfaces * layer
//
// Range: [0, 18*nfaces).
// -----------------------------------------------------------------------
inline long long encodeCorner(int face, int corner, int layer, int nfaces) {
    return (long long)corner + 3LL * face + 3LL * (long long)nfaces * layer;
}

// -----------------------------------------------------------------------
// Build the branched cover from 3-RoSy per-edge permutations.
//
// The `Pcover` vector must be indexed by interior-edge position
// (same order as m.eInt), and each entry is the 6×6 matrix returned
// by augmentPs.
// -----------------------------------------------------------------------
inline CoverBuildResult buildBranchedCover(
    const Mesh& base,
    const std::vector<FaceFrame>& baseFrames,
    const std::vector<CoverPerm>& Pcover,
    const std::vector<int>& singularFields /* per base vertex, bitmask bits 0..2 */) {

    (void)baseFrames;  // parallel transport is baked into Pcover already.
    CoverBuildResult R;
    const int nV = base.nV();
    const int nF = base.nF();
    const int nCover = 6;
    const int nCornerNodes = 3 * nF * nCover;

    // ------------------------------------------------------------------
    // 1. Build adjacency list over corner-layer nodes.
    //
    // For each interior edge (fi, fj), with the 6×6 Pcover[e] (in the
    // reference convention: Pcover(i,j)=1 means "the i-th layer on E(e,0)
    // corresponds to the j-th layer on E(e,1)"), we glue both endpoints
    // of the edge.
    //
    // The edge has two endpoint vertices v1,v2 (its directed endpoints).
    // Each endpoint appears as a local corner on fi and also as a local
    // corner on fj. The gluing connects:
    //
    //   (v*, fi, l_fi)  ↔  (v*, fj, l_fj)   where Pcover(l_fi, l_fj) = 1
    //
    // for both * = 1 and 2.  Exactly matches Weave.cpp:716-746.
    // ------------------------------------------------------------------
    reportProgress(STAGE_COVER_BUILD, 0, 4, "Cover: build gluing adjacency");

    std::vector<std::vector<long long>> adj(nCornerNodes);
    // Reserve a sensible amount of space per node (~2-3 neighbors average).
    for (auto& a : adj) a.reserve(3);

    auto addLink = [&](long long a, long long b) {
        adj[a].push_back(b);
        adj[b].push_back(a);
    };

    for (int e = 0; e < (int)base.eInt.size(); ++e) {
        int h0 = base.eInt[e].first;
        int h1 = base.eInt[e].second;
        int fi = base.he[h0].face;     // = E(e, 0)
        int fj = base.he[h1].face;     // = E(e, 1)
        if (fi < 0 || fj < 0) continue;

        // Endpoints of the undirected edge (the 2 base vertices it
        // touches).  In each face, find the local corner index 0..2
        // for each endpoint vertex.
        int va = base.heStart(h0);     // origin of h0
        int vb = base.he[h0].vertex;   // target of h0

        int aIn_fi = -1, bIn_fi = -1, aIn_fj = -1, bIn_fj = -1;
        for (int c = 0; c < 3; ++c) {
            if (base.F(fi, c) == va) aIn_fi = c;
            if (base.F(fi, c) == vb) bIn_fi = c;
            if (base.F(fj, c) == va) aIn_fj = c;
            if (base.F(fj, c) == vb) bIn_fj = c;
        }
        if (aIn_fi < 0 || bIn_fi < 0 || aIn_fj < 0 || bIn_fj < 0) continue;

        const CoverPerm& P = Pcover[e];

        // For each (source layer on fi = l_fi, target layer on fj = l_fj)
        // where P(l_fi, l_fj) == 1, glue both endpoints (va, vb) of the
        // shared edge.
        for (int l_fi = 0; l_fi < nCover; ++l_fi) {
            int l_fj = -1;
            for (int l = 0; l < nCover; ++l) {
                if (P(l_fi, l) == 1) { l_fj = l; break; }
            }
            if (l_fj < 0) continue;

            long long a_fi = encodeCorner(fi, aIn_fi, l_fi, nF);
            long long b_fi = encodeCorner(fi, bIn_fi, l_fi, nF);
            long long a_fj = encodeCorner(fj, aIn_fj, l_fj, nF);
            long long b_fj = encodeCorner(fj, bIn_fj, l_fj, nF);
            addLink(a_fi, a_fj);
            addLink(b_fi, b_fj);
        }
    }

    // ------------------------------------------------------------------
    // 2. BFS the adjacency graph to find connected components — each is
    // one cover vertex.  Matches Weave.cpp:749-783.
    //
    // We build two mappings:
    //   node2cv[encodedId] = cover-vertex index
    //   cvBase / cvLayer   = per cover vertex, its base vertex and
    //                        "representative layer" (the layer of the
    //                        arbitrarily-chosen first node of the BFS).
    //
    // The representative layer is only used for diagnostics; paired
    // rounding looks up per-layer cover vertices via flatBaseIdx below,
    // which we build with a per-vertex canonical walk.
    // ------------------------------------------------------------------
    reportProgress(STAGE_COVER_BUILD, 1, 4, "Cover: BFS-glue cover vertices");

    std::vector<int> node2cv(nCornerNodes, -1);
    std::vector<int> cvBase, cvLayer;
    cvBase.reserve(nCornerNodes);
    cvLayer.reserve(nCornerNodes);

    for (long long seed = 0; seed < nCornerNodes; ++seed) {
        if (node2cv[seed] != -1) continue;
        int cvId = (int)cvBase.size();
        // Decode seed for base-vertex/layer tag.
        int seedLayer = (int)(seed / (3LL * nF));
        long long rem = seed - (long long)seedLayer * 3LL * nF;
        int seedFace  = (int)(rem / 3);
        int seedCorn  = (int)(rem - 3LL * seedFace);
        int seedBaseV = base.F(seedFace, seedCorn);
        cvBase.push_back(seedBaseV);
        cvLayer.push_back(seedLayer);

        std::queue<long long> q;
        q.push(seed);
        node2cv[seed] = cvId;
        while (!q.empty()) {
            long long cur = q.front(); q.pop();
            for (long long nb : adj[cur]) {
                if (node2cv[nb] != -1) continue;
                node2cv[nb] = cvId;
                q.push(nb);
            }
        }
    }
    const int nCV = (int)cvBase.size();

    // ------------------------------------------------------------------
    // 3. Build flatBaseIdx: per non-singular base vertex v, and each
    // layer k ∈ {0..5}, the cover vertex index.
    //
    // At a non-singular vertex, the monodromy around v is trivial (by
    // construction of "non-singular"), so picking any one incident face
    // f and sampling the corner-node (v, f, layer k) produces the same
    // cvId regardless of which incident face we choose.  We just find
    // one.
    //
    // For singular vertices, leave flatBaseIdx as -1 for every layer,
    // and also mark isSingularBaseV[v] = 1. Paired rounding skips them.
    // ------------------------------------------------------------------
    reportProgress(STAGE_COVER_BUILD, 2, 4, "Cover: flatBaseIdx + singular marks");

    R.isSingularBaseV.assign(nV, 0);
    R.numSingular = 0;
    R.flatBaseIdx.assign((size_t)6 * nV, -1);

    // For each base vertex v, find any incident face (the "seed" face)
    // and local corner index c such that F(seed, c) == v.
    std::vector<int> vSeedFace(nV, -1);
    std::vector<int> vSeedCorn(nV, -1);
    for (int f = 0; f < nF; ++f) {
        for (int c = 0; c < 3; ++c) {
            int v = base.F(f, c);
            if (vSeedFace[v] == -1) {
                vSeedFace[v] = f;
                vSeedCorn[v] = c;
            }
        }
    }

    for (int v = 0; v < nV; ++v) {
        int mask = singularFields[v];
        if (mask != 0) {
            // Any singular field at v: we skip paired rounding at v
            // entirely, since the layer-to-antipodal-layer bijection
            // breaks (non-trivial monodromy merges some of the 6 layers
            // into the same cover vertex).  The reference's mixed-
            // integer rounding handles this gracefully; our simpler
            // Gauss-Seidel code is safer if we just skip these vertices.
            R.isSingularBaseV[v] = 1;
            R.numSingular++;
            continue;
        }
        int seedF = vSeedFace[v];
        if (seedF < 0) continue;
        int seedC = vSeedCorn[v];
        for (int k = 0; k < 6; ++k) {
            long long id = encodeCorner(seedF, seedC, k, nF);
            R.flatBaseIdx[6 * v + k] = node2cv[id];
        }
    }

    // ------------------------------------------------------------------
    // 4. Emit cover faces.
    //
    // For every base face f and every layer k ∈ {0..5}, emit a cover
    // triangle whose 3 vertices are node2cv[encodeCorner(f, c, k)],
    // c = 0..2 — unless any of the 3 corners correspond to a punctured
    // (vertex, layer) pair.
    //
    // Reference: Weave.cpp:785-796 + singularity-removal loop at
    // Weave.cpp:841-855.
    // ------------------------------------------------------------------
    reportProgress(STAGE_COVER_BUILD, 3, 4, "Cover: emit faces + half-edges");

    std::vector<Eigen::Vector3i> CF;
    std::vector<int> fbase, flayer;
    CF.reserve(6 * nF);
    fbase.reserve(6 * nF);
    flayer.reserve(6 * nF);

    for (int f = 0; f < nF; ++f) {
        int v[3] = { base.F(f,0), base.F(f,1), base.F(f,2) };
        int masks[3] = { singularFields[v[0]], singularFields[v[1]], singularFields[v[2]] };

        for (int k = 0; k < 6; ++k) {
            int field = k % 3;
            // If any of the 3 corners has field j=field punctured, skip.
            if ((masks[0] & (1 << field)) ||
                (masks[1] & (1 << field)) ||
                (masks[2] & (1 << field))) continue;

            long long id0 = encodeCorner(f, 0, k, nF);
            long long id1 = encodeCorner(f, 1, k, nF);
            long long id2 = encodeCorner(f, 2, k, nF);
            int a = node2cv[id0];
            int b = node2cv[id1];
            int c = node2cv[id2];
            if (a < 0 || b < 0 || c < 0) continue;
            CF.emplace_back(a, b, c);
            fbase.push_back(f);
            flayer.push_back(k);
        }
    }

    // Cover vertex positions = base vertex positions.
    MatV CV(nCV, 3);
    for (int i = 0; i < nCV; ++i) CV.row(i) = base.V.row(cvBase[i]);

    R.mesh.V = CV;
    R.mesh.F.resize((int)CF.size(), 3);
    for (int i = 0; i < (int)CF.size(); ++i) R.mesh.F.row(i) = CF[i];
    buildHalfEdges(R.mesh);

    R.coverFaceBaseF   = std::move(fbase);
    R.coverFaceLayer   = std::move(flayer);
    R.coverVertexBaseV = std::move(cvBase);
    R.coverVertexLayer = std::move(cvLayer);

    std::fprintf(stderr,
        "[cover] built: V_base=%d F_base=%d | cover V=%d F=%d | singular base V=%d (%.1f%%)\n",
        nV, nF, R.mesh.nV(), R.mesh.nF(), R.numSingular,
        100.0 * R.numSingular / std::max(1, nV));

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
