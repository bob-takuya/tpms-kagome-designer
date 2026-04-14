// rosy.h — 6-RoSy globally smooth direction field (Knöppel et al. 2013),
// used as the initial ŵ⁰ for Algorithm 1 of Vekhter 2019.
//
// The 6-RoSy Dirichlet energy on per-face complex scalars z_f = e^{6 i φ_f}
// is
//                                                 2
//     E(z) = Σ w_e | z_j − e^{6 i β_ij} z_i |
//             e
//
// where β_ij is the parallel-transport rotation from fi to fj across
// interior edge e. The smallest generalized eigenvector of the
// corresponding Hermitian matrix L (vs a face-area mass M) gives the
// globally optimal direction field.
//
// Inverse power iteration with shift σ = 1e-4 (on (L + σM)) is used to
// drive to the smallest eigenvector; each inner solve is a complex
// Cholesky (L_Chol = SimplicialLDLT on the real 2×2 block form).

#pragma once

#include "mesh.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <complex>

namespace wgf {

using Cplx = std::complex<double>;

// Build the 6-RoSy connection-Laplacian as a real 2n×2n symmetric PSD
// matrix by representing each complex entry (a + i b) as a 2×2 block
// [[a, -b], [b, a]].  This lets us use Eigen's real SimplicialLDLT.
inline SpMat build6RoSyReal(const Mesh& m,
                            const std::vector<FaceFrame>& frames) {
    const int F = m.nF();
    std::vector<Trip> T;
    T.reserve(m.eInt.size() * 16);

    for (const auto& e : m.eInt) {
        int h = e.first;
        int t = e.second;
        int fi = m.he[h].face;
        int fj = m.he[t].face;
        double w = std::max(0.0, cotWeight(m, h));

        double beta = parallelTransport(m, frames, h);
        double theta = 6.0 * beta;
        double c = std::cos(theta), s = std::sin(theta);

        int ii = 2 * fi, jj = 2 * fj;
        // diagonals: + w I on both (i,i) and (j,j)
        T.emplace_back(ii,   ii,   w);
        T.emplace_back(ii+1, ii+1, w);
        T.emplace_back(jj,   jj,   w);
        T.emplace_back(jj+1, jj+1, w);
        // off-diagonals: −w R(θ) on (j,i) block
        T.emplace_back(jj,   ii,   -w * c);
        T.emplace_back(jj,   ii+1,  w * s);
        T.emplace_back(jj+1, ii,   -w * s);
        T.emplace_back(jj+1, ii+1, -w * c);
        // symmetric (i,j) block: (R(θ))ᵀ = R(−θ)
        T.emplace_back(ii,   jj,   -w * c);
        T.emplace_back(ii,   jj+1, -w * s);
        T.emplace_back(ii+1, jj,    w * s);
        T.emplace_back(ii+1, jj+1, -w * c);
    }
    SpMat L(2 * F, 2 * F);
    L.setFromTriplets(T.begin(), T.end());
    return L;
}

// Per-face mass matrix (2F × 2F diagonal) — two copies of the face area.
inline Vec face6RoSyMass(const std::vector<FaceFrame>& frames) {
    return faceFieldMass(frames);
}

// Run inverse power iteration on (L + σM) z = λ M z to get the
// smallest generalized eigenvector. Returns a 2F vector that encodes
// per-face complex numbers z_f = (re_f, im_f). The canonical 6-RoSy
// angle is φ_f = atan2(im_f, re_f) / 6.
inline Vec solve6RoSy(const SpMat& L,
                      const Vec& M,
                      int nOuter = 20,
                      double shift = 1e-4) {
    const int n = (int)L.rows();
    // Build A = L + shift * diag(M)
    SpMat Mdiag(n, n);
    std::vector<Trip> Tm;
    Tm.reserve(n);
    for (int i = 0; i < n; ++i) Tm.emplace_back(i, i, M[i]);
    Mdiag.setFromTriplets(Tm.begin(), Tm.end());
    SpMat A = L + shift * Mdiag;
    A.makeCompressed();

    Eigen::SimplicialLDLT<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) {
        // degenerate case: fall back to identity
        return Vec::Zero(n);
    }

    // Random init, area-normalized.
    Vec x = Vec::Random(n);
    auto normalize = [&](Vec& v) {
        double nr = 0;
        for (int i = 0; i < n; ++i) nr += M[i] * v[i] * v[i];
        v /= std::sqrt(std::max(nr, 1e-30));
    };
    normalize(x);

    for (int it = 0; it < nOuter; ++it) {
        Vec b = M.asDiagonal() * x;
        Vec y = solver.solve(b);
        normalize(y);
        x = y;
    }
    return x;
}

// Extract canonical 6-RoSy per-face angle φ_f ∈ (−π/6, π/6].
inline Vec faceBaseAngleFromZ(const Vec& z) {
    const int F = (int)z.size() / 2;
    Vec phi(F);
    for (int f = 0; f < F; ++f) {
        phi[f] = std::atan2(z[2*f+1], z[2*f]) / 6.0;
    }
    return phi;
}

// Lift per-face 6-RoSy angle to a 2|F|-vector of unit direction
// vectors (layer 0 of the RoSy — used to initialize Algorithm 1).
inline Vec initFaceFieldFromRosy(const Vec& phi) {
    const int F = (int)phi.size();
    Vec w(2 * F);
    for (int f = 0; f < F; ++f) {
        w[2*f]   = std::cos(phi[f]);
        w[2*f+1] = std::sin(phi[f]);
    }
    return w;
}

} // namespace wgf
