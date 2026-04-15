// Native (non-Emscripten) smoke test for the WGF C++ implementation.
// Synthesizes a small Schwarz-P surface via marching cubes in-place and
// runs Knöppel 2013 + Algorithm 1 to verify the solver stack compiles
// and produces sensible numbers.
//
// Build:
//   cd native/build
//   cmake -DCMAKE_BUILD_TYPE=Release .. && make smoke_native
//   ./smoke_native

#include "mesh.h"
#include "rosy.h"
#include "alg1.h"
#include "curl.h"
#include "cover.h"
#include "permutations.h"
#include "alg2.h"
#include "pipeline.h"

#include <map>
#include <tuple>

#include <cstdio>
#include <cmath>

using namespace wgf;

// Very small, hand-rolled marching cubes just for the smoke test.
// Produces a triangle mesh of {f(x,y,z) = 0}.
namespace {

static const int EDGE_TABLE[12][2] = {
    {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}
};

static const int TRI_TABLE[256][16] = {
    // Use the standard Paul Bourke table; truncated here to −1 terminators
    // via a preprocessor-generated file at build time. For the smoke test
    // we inline an extremely simple alternative: sample the implicit
    // function at tetrahedra instead of cubes. See below.
    {0}
};

// Marching tetrahedra — simpler than cubes, sufficient for a smoke test.
struct Tri { double a[3], b[3], c[3]; };

inline void interp(double out[3], const double p[3], const double q[3],
                   double vp, double vq) {
    double t = vp / (vp - vq);
    out[0] = p[0] + t * (q[0] - p[0]);
    out[1] = p[1] + t * (q[1] - p[1]);
    out[2] = p[2] + t * (q[2] - p[2]);
}

template <class Fn>
void marchTet(Fn f, double p[4][3], std::vector<Tri>& tris) {
    double vals[4] = { f(p[0][0],p[0][1],p[0][2]),
                       f(p[1][0],p[1][1],p[1][2]),
                       f(p[2][0],p[2][1],p[2][2]),
                       f(p[3][0],p[3][1],p[3][2]) };
    int mask = 0;
    for (int i = 0; i < 4; ++i) if (vals[i] < 0) mask |= (1 << i);
    // 16 cases, 14 useful. Use a compact switch-table.
    auto out3 = [&](int a, int b, int c, int d, int e, int fi) {
        Tri t;
        interp(t.a, p[a], p[b], vals[a], vals[b]);
        interp(t.b, p[c], p[d], vals[c], vals[d]);
        interp(t.c, p[e], p[fi], vals[e], vals[fi]);
        tris.push_back(t);
    };
    auto out4 = [&](int a, int b, int c, int d, int e, int fi, int g, int h) {
        Tri t1, t2;
        interp(t1.a, p[a], p[b], vals[a], vals[b]);
        interp(t1.b, p[c], p[d], vals[c], vals[d]);
        interp(t1.c, p[e], p[fi], vals[e], vals[fi]);
        tris.push_back(t1);
        interp(t2.a, p[c], p[d], vals[c], vals[d]);
        interp(t2.b, p[g], p[h], vals[g], vals[h]);
        interp(t2.c, p[e], p[fi], vals[e], vals[fi]);
        tris.push_back(t2);
    };
    switch (mask) {
        case 0x0: case 0xF: break;
        case 0x1: out3(0,1, 0,2, 0,3); break;
        case 0xE: out3(0,1, 0,3, 0,2); break;
        case 0x2: out3(1,0, 1,3, 1,2); break;
        case 0xD: out3(1,0, 1,2, 1,3); break;
        case 0x4: out3(2,0, 2,1, 2,3); break;
        case 0xB: out3(2,0, 2,3, 2,1); break;
        case 0x8: out3(3,0, 3,2, 3,1); break;
        case 0x7: out3(3,0, 3,1, 3,2); break;
        case 0x3: out4(0,2, 1,2, 0,3, 1,3); break;
        case 0xC: out4(0,2, 0,3, 1,2, 1,3); break;
        case 0x5: out4(0,1, 2,1, 0,3, 2,3); break;
        case 0xA: out4(0,1, 0,3, 2,1, 2,3); break;
        case 0x6: out4(0,1, 0,2, 1,3, 2,3); break;
        case 0x9: out4(0,1, 1,3, 0,2, 2,3); break;
    }
}

template <class Fn>
void marchGrid(Fn f, double lo, double hi, int res, std::vector<Tri>& tris) {
    double step = (hi - lo) / res;
    // 5-tetrahedra decomposition of a cube
    static const int TETS[5][4] = {
        {0,1,2,5}, {0,2,7,5}, {0,2,3,7}, {0,5,7,4}, {2,5,7,6}
    };
    for (int i = 0; i < res; ++i)
    for (int j = 0; j < res; ++j)
    for (int k = 0; k < res; ++k) {
        double cx = lo + i * step;
        double cy = lo + j * step;
        double cz = lo + k * step;
        double C[8][3] = {
            {cx,        cy,        cz       },
            {cx + step, cy,        cz       },
            {cx + step, cy + step, cz       },
            {cx,        cy + step, cz       },
            {cx,        cy,        cz + step},
            {cx + step, cy,        cz + step},
            {cx + step, cy + step, cz + step},
            {cx,        cy + step, cz + step}
        };
        for (int t = 0; t < 5; ++t) {
            double P[4][3] = {
                { C[TETS[t][0]][0], C[TETS[t][0]][1], C[TETS[t][0]][2] },
                { C[TETS[t][1]][0], C[TETS[t][1]][1], C[TETS[t][1]][2] },
                { C[TETS[t][2]][0], C[TETS[t][2]][1], C[TETS[t][2]][2] },
                { C[TETS[t][3]][0], C[TETS[t][3]][1], C[TETS[t][3]][2] }
            };
            marchTet(f, P, tris);
        }
    }
}

// Build a guaranteed-manifold icosphere by recursively subdividing an
// icosahedron. Every subdivision step replaces each triangle with 4
// sub-triangles, keeping vertex sharing along edges exact (each new
// midpoint is keyed by the ordered edge endpoint indices, not by 3D
// coordinates). Produces meshes of known topology (genus 0, χ=2).
inline Mesh buildIcosphere(int subdivisions) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<Eigen::Vector3d> V = {
        {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
        { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
        { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1},
    };
    for (auto& p : V) p.normalize();

    std::vector<Eigen::Vector3i> F = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
    };

    for (int s = 0; s < subdivisions; ++s) {
        std::map<std::pair<int,int>, int> mid;
        auto getMid = [&](int a, int b) -> int {
            auto k = std::make_pair(std::min(a,b), std::max(a,b));
            auto it = mid.find(k);
            if (it != mid.end()) return it->second;
            Eigen::Vector3d m = (V[a] + V[b]).normalized();
            int id = (int)V.size();
            V.push_back(m);
            mid[k] = id;
            return id;
        };
        std::vector<Eigen::Vector3i> F2;
        F2.reserve(F.size() * 4);
        for (const auto& f : F) {
            int a = f[0], b = f[1], c = f[2];
            int ab = getMid(a,b), bc = getMid(b,c), ca = getMid(c,a);
            F2.push_back({a,  ab, ca});
            F2.push_back({b,  bc, ab});
            F2.push_back({c,  ca, bc});
            F2.push_back({ab, bc, ca});
        }
        F = std::move(F2);
    }

    Mesh m;
    m.V.resize((int)V.size(), 3);
    for (int i = 0; i < (int)V.size(); ++i) m.V.row(i) = V[i];
    m.F.resize((int)F.size(), 3);
    for (int i = 0; i < (int)F.size(); ++i) m.F.row(i) = F[i];
    buildHalfEdges(m);
    return m;
}

Mesh trisToMesh(const std::vector<Tri>& tris, double weldEps = 1e-6) {
    // Proper 3D-coordinate weld via std::map<tuple,int>. The earlier
    // 1D-hash version collapsed distinct points together (creating fake
    // non-manifold vertices) AND failed to merge coincident ones
    // (creating spurious boundaries) — tripling the singular-vertex
    // count reported by the cover build.
    std::map<std::tuple<long long, long long, long long>, int> uniq;
    std::vector<Eigen::Vector3d> V;
    std::vector<Eigen::Vector3i> F;
    auto snap = [&](double x) -> long long {
        return (long long)std::llround(x / weldEps);
    };
    auto insert = [&](const double p[3]) -> int {
        auto k = std::make_tuple(snap(p[0]), snap(p[1]), snap(p[2]));
        auto it = uniq.find(k);
        if (it != uniq.end()) return it->second;
        int id = (int)V.size();
        V.emplace_back(p[0], p[1], p[2]);
        uniq[k] = id;
        return id;
    };
    for (const auto& t : tris) {
        int a = insert(t.a);
        int b = insert(t.b);
        int c = insert(t.c);
        if (a == b || b == c || a == c) continue;
        F.emplace_back(a, b, c);
    }
    Mesh m;
    m.V.resize((int)V.size(), 3);
    for (int i = 0; i < (int)V.size(); ++i) m.V.row(i) = V[i];
    m.F.resize((int)F.size(), 3);
    for (int i = 0; i < (int)F.size(); ++i) m.F.row(i) = F[i];
    buildHalfEdges(m);
    return m;
}

} // anonymous

int main() {
    // Icosphere — guaranteed manifold (genus 0, χ=2). 4 subdivisions
    // → 2562 vertices / 5120 faces. Used as a clean baseline for
    // validating the WGF pipeline; the earlier marching-tetrahedra
    // test mesh was non-manifold due to tet-face coincident-point
    // floating-point mismatches, which confused the cover build.
    Mesh m = buildIcosphere(4);
    std::printf("[smoke] icosphere mesh: V=%d  F=%d  E_int=%d  "
                "(3F - 2E_int = %d boundary half-edges, expect 0)\n",
                m.nV(), m.nF(), m.nE(), 3*m.nF() - 2*m.nE());

    auto frames = buildFaceFrames(m);

    // 6-RoSy
    SpMat L = build6RoSyReal(m, frames);
    Vec   M = face6RoSyMass(frames);
    Vec   z = solve6RoSy(L, M, 25, 1e-4);
    Vec   phi = faceBaseAngleFromZ(z);
    Vec   w0  = initFaceFieldFromRosy(phi);

    // Algorithm 1 (paper-strict with λ sharpening)
    Alg1Result R = runAlg1(m, frames, w0,
                           /*handles=*/{},
                           /*lambdaInit=*/1000.0,
                           /*lambdaMin=*/ 1e-3,
                           /*maxIter=*/   50,
                           /*tol=*/       1e-5);
    std::printf("[smoke] Algorithm 1: curl %.3e -> %.3e  (%.1f%% reduction)  in %d iters\n",
                R.initialCurl, R.finalCurl,
                (1.0 - R.finalCurl / std::max(R.initialCurl, 1e-30)) * 100.0,
                R.iterations);
    std::printf("[smoke] λ schedule: %zu steps\n", R.lambdaHistory.size());

    // Debug: histogram of phi
    {
        double mn = 1e30, mx = -1e30;
        for (int i = 0; i < phi.size(); ++i) {
            mn = std::min(mn, phi[i]);
            mx = std::max(mx, phi[i]);
        }
        std::printf("[smoke] phi range: [%.4f, %.4f] rad  (expected ~[-π/6, π/6] = [%.4f, %.4f])\n",
                    mn, mx, -M_PI/6, M_PI/6);
    }

    // ── 6-fold branched cover (Vekhter 2019 §5.1 faithful port) ──────────
    //
    // Reference pipeline:
    //   splitFromRoSy → 3-RoSy vectors per face
    //   reassignAllPermutations → per-edge 3×3 signed permutations
    //   findSingularFieldsPerVertex → per-field vertex singularities
    //   augmentPs → 6×6 cover permutations
    //   buildBranchedCover → BFS-glued cover mesh
    Vec wRep3 = splitFromRoSy(m, frames, w0);
    auto Ps = reassignAllPermutations(m, frames, wRep3);
    {
        PsHistogram H = histogramPs(Ps);
        std::printf("[smoke] Ps histogram: identity=%d sign-only=%d permuting=%d / %d edges\n",
                    H.identity, H.signOnly, H.permuting, (int)Ps.size());
    }
    auto singularFields = findSingularFieldsPerVertex(m, Ps);
    {
        int nSing = 0, nSingFields = 0;
        for (int mask : singularFields) {
            if (mask) {
                nSing++;
                for (int b = 0; b < 3; ++b) if (mask & (1 << b)) nSingFields++;
            }
        }
        std::printf("[smoke] singular verts=%d (v,field)-singularities=%d\n",
                    nSing, nSingFields);
    }
    auto Pcover = augmentPs(Ps);
    auto cov = buildBranchedCover(m, frames, Pcover, singularFields);
    std::printf("[smoke] cover:   V=%d  F=%d  singular=%d (%.1f%%)\n",
                cov.mesh.nV(), cov.mesh.nF(), cov.numSingular,
                100.0 * cov.numSingular / std::max(1, m.nV()));

    std::vector<int> comp;
    int nComp = labelComponents(cov.mesh, comp);
    std::printf("[smoke] cover connected components: %d\n", nComp);

    // End-to-end pipeline run.
    {
        PipelineOptions opt;
        opt.useCover = true;    // now exercises the 3-family rotated-Knoeppel path
        opt.alg1MaxIter = 30;
        PipelineResult R2 = runPipeline(m, opt);
        std::printf("[smoke/3-family] segments=%zu (fam %d/%d/%d) curl %.3e -> %.3e  in %d iters\n",
                    R2.segments.size(),
                    R2.numSegmentsFam[0], R2.numSegmentsFam[1], R2.numSegmentsFam[2],
                    R2.alg1BaseInitCurl, R2.alg1BaseFinalCurl, R2.alg1BaseIters);
    }

    // Run Alg1 on each component.
    auto covFrames = buildFaceFrames(cov.mesh);
    // Lift the base 6-RoSy φ to the cover by picking layer k per cover face.
    {
        Vec covW(2 * cov.mesh.nF());
        for (int cf = 0; cf < cov.mesh.nF(); ++cf) {
            int bf  = cov.coverFaceBaseF[cf];
            int lay = cov.coverFaceLayer[cf];
            double ang = phi[bf] + lay * M_PI / 3.0;
            // Express in cover face's frame, which differs from base
            // face frame because vertex ordering may change during
            // sub-meshing — transport to the cover frame.
            // Base direction in 3D:
            Eigen::Vector3d d3 =
                std::cos(ang) * frames[bf].e1 +
                std::sin(ang) * frames[bf].e2;
            double u = d3.dot(covFrames[cf].e1);
            double v = d3.dot(covFrames[cf].e2);
            double L = std::sqrt(u*u + v*v);
            if (L < 1e-12) { u = 1; v = 0; L = 1; }
            covW[2*cf]     = u / L;
            covW[2*cf + 1] = v / L;
        }
        Alg1Result R2 = runAlg1(cov.mesh, covFrames, covW, {},
                                1000.0, 1e-3, 50, 1e-5);
        std::printf("[smoke] cover Alg1: curl %.3e -> %.3e  in %d iters\n",
                    R2.initialCurl, R2.finalCurl, R2.iterations);
    }

    return 0;
}
