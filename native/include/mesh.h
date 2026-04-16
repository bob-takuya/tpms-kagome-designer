// mesh.h — half-edge triangle mesh, face frames, cotangent Laplacian.
//
// Lightweight data structures used by the Weaving Geodesic Foliations
// implementation (Vekhter et al., SIGGRAPH 2019).  No third-party
// dependencies except Eigen.
//
// Conventions:
//   - V:   |V| × 3 row-major matrix of vertex positions.
//   - F:   |F| × 3 row-major matrix of triangle vertex indices.
//   - E_int: list of interior (non-boundary) undirected edges, each
//            stored as a pair of directed half-edges (he, twin(he)).
//   - FaceFrame: per-face orthonormal basis (e1, e2, n) with
//                e1 = (v1 − v0)/‖v1 − v0‖, n = e1 × (v2 − v0) normalized,
//                e2 = n × e1.
//
// This is pure header code so that individual compilation units can
// pick the pieces they need.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <cassert>

namespace wgf {

using MatV = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatF = Eigen::Matrix<int,    Eigen::Dynamic, 3, Eigen::RowMajor>;
using Vec  = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;
using Trip  = Eigen::Triplet<double>;

struct HalfEdge {
    int vertex;   // vertex this half-edge points TO
    int face;     // face to the left of the half-edge
    int next;     // next half-edge in the face (CCW)
    int prev;     // prev half-edge in the face (CCW)
    int twin;     // opposite half-edge (-1 if boundary)
};

struct Mesh {
    MatV V;                         // |V| × 3 positions
    MatF F;                         // |F| × 3 triangles
    std::vector<HalfEdge> he;       // 3·|F| half-edges
    std::vector<int> v2he;          // some outgoing half-edge per vertex
    std::vector<int> f2he;          // first half-edge per face
    // Interior edge list: (he_a, he_b) with he_a < he_b, he_b = twin(he_a).
    std::vector<std::pair<int,int>> eInt;

    int nV() const { return (int)V.rows(); }
    int nF() const { return (int)F.rows(); }
    int nE() const { return (int)eInt.size(); }

    int heStart(int h) const { return he[he[h].prev].vertex; }
};

struct FaceFrame {
    Eigen::Vector3d e1, e2, n;
    double area;
};

inline void buildHalfEdges(Mesh& m) {
    const int F = m.nF();
    m.he.clear();
    m.he.reserve(3 * F);
    m.f2he.assign(F, -1);
    m.v2he.assign(m.nV(), -1);

    // Edge map: key = va*|V|+vb with va<vb.
    struct Key { long long k; };
    std::unordered_map<long long, int> edgeMap;
    edgeMap.reserve(3 * F);

    for (int f = 0; f < F; ++f) {
        int v[3] = { m.F(f,0), m.F(f,1), m.F(f,2) };
        int base = (int)m.he.size();
        m.f2he[f] = base;
        for (int i = 0; i < 3; ++i) {
            HalfEdge h{};
            h.vertex = v[(i+1)%3];
            h.face   = f;
            h.next   = base + (i+1)%3;
            h.prev   = base + (i+2)%3;
            h.twin   = -1;
            m.he.push_back(h);
        }
        for (int i = 0; i < 3; ++i) {
            int heIdx = base + i;
            int a = v[i], b = v[(i+1)%3];
            if (m.v2he[a] < 0) m.v2he[a] = heIdx;
            long long key = (long long)std::min(a,b) * (long long)m.nV() + std::max(a,b);
            auto it = edgeMap.find(key);
            if (it == edgeMap.end()) {
                edgeMap[key] = heIdx;
            } else {
                m.he[heIdx].twin = it->second;
                m.he[it->second].twin = heIdx;
            }
        }
    }

    // Interior edges.
    m.eInt.clear();
    for (int h = 0; h < (int)m.he.size(); ++h) {
        int t = m.he[h].twin;
        if (t < 0) continue;
        if (h < t) m.eInt.emplace_back(h, t);
    }
}

inline std::vector<FaceFrame> buildFaceFrames(const Mesh& m) {
    const int nF = m.nF();
    std::vector<FaceFrame> out(nF);
    for (int f = 0; f < nF; ++f) {
        Eigen::Vector3d a = m.V.row(m.F(f,0));
        Eigen::Vector3d b = m.V.row(m.F(f,1));
        Eigen::Vector3d c = m.V.row(m.F(f,2));
        Eigen::Vector3d ab = b - a;
        Eigen::Vector3d ac = c - a;
        Eigen::Vector3d nn = ab.cross(ac);
        double len = nn.norm();
        out[f].area = 0.5 * len;
        out[f].n  = nn / std::max(len, 1e-30);
        out[f].e1 = ab.normalized();
        out[f].e2 = out[f].n.cross(out[f].e1).normalized();
    }
    return out;
}

// Return cotangent weight of an interior half-edge (sum of the two
// opposite cotangents across the edge).
inline double cotWeight(const Mesh& m, int h) {
    int t = m.he[h].twin;
    if (t < 0) return 0.0;
    // Vertices of the edge:
    int vi = m.heStart(h);
    int vj = m.he[h].vertex;
    // Opposite vertices:
    int vk1 = m.he[m.he[h].next].vertex;
    int vk2 = m.he[m.he[t].next].vertex;
    auto cot = [&](int A, int B, int C) {
        Eigen::Vector3d a = m.V.row(A);
        Eigen::Vector3d b = m.V.row(B);
        Eigen::Vector3d c = m.V.row(C);
        Eigen::Vector3d ba = a - b;
        Eigen::Vector3d bc = c - b;
        double dot = ba.dot(bc);
        double cr  = ba.cross(bc).norm();
        return cr > 1e-30 ? dot / cr : 0.0;
    };
    return 0.5 * (cot(vi, vk1, vj) + cot(vi, vk2, vj));
}

// Vertex cotangent Laplacian (Laplace–Beltrami).
//
// L[i,i] = Σ_j w_ij        (positive-semidefinite convention)
// L[i,j] = −w_ij
inline SpMat buildVertexCotanL(const Mesh& m) {
    const int n = m.nV();
    std::vector<Trip> T;
    T.reserve(m.he.size() * 2);
    for (const auto& e : m.eInt) {
        int h = e.first;
        int i = m.heStart(h);
        int j = m.he[h].vertex;
        double w = std::max(0.0, cotWeight(m, h));
        T.emplace_back(i, i,  w);
        T.emplace_back(j, j,  w);
        T.emplace_back(i, j, -w);
        T.emplace_back(j, i, -w);
    }
    SpMat L(n, n);
    L.setFromTriplets(T.begin(), T.end());
    return L;
}

// Voronoi vertex-area mass matrix (diagonal).
inline Vec vertexMass(const Mesh& m) {
    const int n = m.nV();
    Vec mass = Vec::Zero(n);
    const auto frames = buildFaceFrames(m);
    for (int f = 0; f < m.nF(); ++f) {
        double a = frames[f].area / 3.0;
        mass[m.F(f,0)] += a;
        mass[m.F(f,1)] += a;
        mass[m.F(f,2)] += a;
    }
    return mass;
}

// Face-based mass matrix (2|F| × 2|F| diagonal block, two copies per face).
// Used as Mvf inside Algorithm 1's KKT system.
inline Vec faceFieldMass(const std::vector<FaceFrame>& frames) {
    const int F = (int)frames.size();
    Vec m(2 * F);
    for (int f = 0; f < F; ++f) {
        m[2*f]   = frames[f].area;
        m[2*f+1] = frames[f].area;
    }
    return m;
}

// Face-based cotangent Dirichlet energy for 2D face vector fields.
// Uses the Crane-et-al. "connection-respecting" discretization:
// for each interior edge with parallel-transport rotation β (fi → fj),
// the local smoothness term is w_e ‖u_j − R(β) u_i‖², which gives a
// 4×4 block per edge.
//
// Returns a 2|F| × 2|F| symmetric PSD sparse matrix.
inline SpMat buildFaceDirichlet(const Mesh& m,
                                const std::vector<FaceFrame>& frames) {
    const int F = (int)frames.size();
    std::vector<Trip> T;
    T.reserve(m.eInt.size() * 16);
    for (const auto& e : m.eInt) {
        int h = e.first;
        int t = e.second;
        int fi = m.he[h].face;
        int fj = m.he[t].face;
        double w = std::max(0.0, cotWeight(m, h));

        // Shared edge vector, expressed in each face's local 2D frame.
        Eigen::Vector3d ev = m.V.row(m.he[h].vertex) - m.V.row(m.heStart(h));
        Eigen::Vector2d ei(ev.dot(frames[fi].e1), ev.dot(frames[fi].e2));
        Eigen::Vector2d ej(ev.dot(frames[fj].e1), ev.dot(frames[fj].e2));
        double ai = std::atan2(ei.y(), ei.x());
        double aj = std::atan2(ej.y(), ej.x());
        double beta = aj - ai;     // rotation fi → fj
        double cb = std::cos(beta), sb = std::sin(beta);
        // R = [cb -sb; sb cb]

        // ‖u_j − R u_i‖² = u_jᵀ u_j − 2 u_jᵀ R u_i + u_iᵀ u_i
        int ii = 2 * fi, jj = 2 * fj;
        // diagonal u_iᵀ u_i
        T.emplace_back(ii,   ii,   w);
        T.emplace_back(ii+1, ii+1, w);
        // diagonal u_jᵀ u_j
        T.emplace_back(jj,   jj,   w);
        T.emplace_back(jj+1, jj+1, w);
        // off-diagonal −u_jᵀ R u_i (and its transpose)
        //   −u_j · (R u_i) = −[u_jx, u_jy] · [cb u_ix − sb u_iy, sb u_ix + cb u_iy]
        //                 = −cb u_jx u_ix + sb u_jx u_iy − sb u_jy u_ix − cb u_jy u_iy
        T.emplace_back(jj,   ii,   -w * cb);
        T.emplace_back(jj,   ii+1,  w * sb);
        T.emplace_back(jj+1, ii,   -w * sb);
        T.emplace_back(jj+1, ii+1, -w * cb);
        // symmetric mirror
        T.emplace_back(ii,   jj,   -w * cb);
        T.emplace_back(ii+1, jj,    w * sb);
        T.emplace_back(ii,   jj+1, -w * sb);
        T.emplace_back(ii+1, jj+1, -w * cb);
    }
    SpMat L(2 * F, 2 * F);
    L.setFromTriplets(T.begin(), T.end());
    return L;
}

// Drop triangles whose area is below `areaThreshold`. Marching-cubes can
// emit a few zero-area slivers along voxel boundaries; downstream Algorithm 1
// builds a KKT system whose diagonal includes per-face mass terms, so a single
// degenerate face injects a hard zero on the diagonal and breaks SparseLU.
// Removes faces only — vertices are kept (they may be referenced elsewhere).
// Caller must rebuild half-edges; this routine clears `m.he/v2he/f2he/eInt`
// because their indices reference the old face list.
inline int removeDegenerateFaces(Mesh& m, double areaThreshold = 1e-14) {
    const int nF0 = m.nF();
    std::vector<int> keep;
    keep.reserve(nF0);
    for (int f = 0; f < nF0; ++f) {
        Eigen::Vector3d a = m.V.row(m.F(f, 0));
        Eigen::Vector3d b = m.V.row(m.F(f, 1));
        Eigen::Vector3d c = m.V.row(m.F(f, 2));
        double area = 0.5 * (b - a).cross(c - a).norm();
        if (area > areaThreshold) keep.push_back(f);
    }
    const int nFnew = (int)keep.size();
    if (nFnew == nF0) return 0;
    MatF Fnew(nFnew, 3);
    for (int i = 0; i < nFnew; ++i) Fnew.row(i) = m.F.row(keep[i]);
    m.F = std::move(Fnew);
    m.he.clear();
    m.v2he.clear();
    m.f2he.clear();
    m.eInt.clear();
    return nF0 - nFnew;
}

// Parallel-transport angle from face fi to face fj across their shared
// half-edge h (h ∈ face fi). Needed by the 6-RoSy build.
inline double parallelTransport(const Mesh& m,
                                const std::vector<FaceFrame>& frames,
                                int h) {
    int t = m.he[h].twin;
    assert(t >= 0);
    int fi = m.he[h].face;
    int fj = m.he[t].face;
    Eigen::Vector3d ev = m.V.row(m.he[h].vertex) - m.V.row(m.heStart(h));
    double ai = std::atan2(ev.dot(frames[fi].e2), ev.dot(frames[fi].e1));
    double aj = std::atan2(ev.dot(frames[fj].e2), ev.dot(frames[fj].e1));
    return aj - ai;
}

} // namespace wgf
