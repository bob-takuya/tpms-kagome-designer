// alg1.h — Vekhter et al. 2019, Algorithm 1 (§3.3, eq. 5).
//
// Alternating minimization of
//
//     argmin_{ŵ, δ}  (1/2)‖δ‖²_Mvf + (λ/2)‖∇(ŵ + δ)‖²_Lvf
//          subject to   C(ŵ + δ) = 0,
//                        ‖ŵ_f‖ = 1 ∀ f,
//                        ŵ_f = ŵ_f^H and δ_f = 0 ∀ f ∈ H (handle faces).
//
// At iteration i:
//
//   Step A (δ update — linear KKT system):
//
//     [ Mvf + λ Lvf      Cᵀ ] [ δ ]     [ −λ Lvf ŵ^{i−1} ]
//     [      C            0 ] [ μ ]  =  [   −C  ŵ^{i−1}  ]
//
//     with handle rows/cols replaced by identity and RHS=0.
//
//   Step B (per-face unit normalization):
//
//     ŵ_f^i = (ŵ_f^{i−1} + δ_f) / ‖ŵ_f^{i−1} + δ_f‖
//
//   Residual δ_f^i = (ŵ_f^{i−1} + δ_f) − ŵ_f^i  is reported.
//
// Convergence check: max face-update magnitude < tol.
//
// λ-sharpening: repeatedly halve λ once inner convergence is reached,
// following §5.2 recommendation (1000 → 0).

#pragma once

#include "mesh.h"
#include "curl.h"
#include <Eigen/SparseLU>
#include <vector>

namespace wgf {

struct Alg1Result {
    Vec w;              // final unit vector field (2|F|)
    double initialCurl; // ‖C ŵ⁰‖
    double finalCurl;   // ‖C ŵ‖
    int iterations;
    std::vector<double> lambdaHistory;
};

// Build the face-based Dirichlet energy L_vf (see mesh.h).
// We re-declare here for clarity.
inline SpMat makeLvf(const Mesh& m,
                     const std::vector<FaceFrame>& frames) {
    return buildFaceDirichlet(m, frames);
}

// Apply handle constraints: zero rows/cols of the KKT block corresponding
// to handle DOFs, put identity on the diagonal, and zero the RHS.
inline void applyHandles(SpMat& K, Vec& rhs,
                         const std::vector<int>& handleFaces,
                         int nW) {
    if (handleFaces.empty()) return;
    // Build a set of handle DOF indices.
    std::vector<char> isHandle(nW, 0);
    for (int f : handleFaces) {
        isHandle[2*f]     = 1;
        isHandle[2*f + 1] = 1;
    }
    // Zero-out matrix rows/cols incident to handles; set identity on the
    // diagonal. Working on a copy avoids modifying iterators mid-pass.
    SpMat K2 = K;
    for (int k = 0; k < K2.outerSize(); ++k) {
        for (SpMat::InnerIterator it(K2, k); it; ++it) {
            int r = (int)it.row();
            int c = (int)it.col();
            if (r < nW && isHandle[r]) K.coeffRef(r, c) = (r == c) ? 1.0 : 0.0;
            if (c < nW && isHandle[c]) K.coeffRef(r, c) = (r == c) ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < nW; ++i) if (isHandle[i]) { K.coeffRef(i, i) = 1.0; rhs[i] = 0.0; }
    K.prune(1e-30);
}

inline Alg1Result runAlg1(const Mesh& m,
                          const std::vector<FaceFrame>& frames,
                          const Vec& w0,
                          const std::vector<int>& handleFaces = {},
                          double lambdaInit = 1000.0,
                          double lambdaMin  = 1e-3,
                          int    maxIter    = 50,
                          double tol        = 1e-5) {
    const int F = m.nF();
    const int nW = 2 * F;
    const int E  = m.nE();

    SpMat C   = buildCurlOperator(m, frames);
    SpMat Ct  = SpMat(C.transpose());
    Vec   Mvf = faceFieldMass(frames);
    SpMat Lvf = makeLvf(m, frames);

    // Build the base KKT pattern once; we'll update the upper-left block
    // each time λ changes.
    //
    //   K = [ A(λ)   Cᵀ ]        A(λ) = diag(Mvf) + λ Lvf
    //       [   C    0  ]
    //
    auto buildK = [&](double lambda) -> SpMat {
        std::vector<Trip> T;
        T.reserve(Lvf.nonZeros() + nW + C.nonZeros() * 2);
        // A(λ): diag(Mvf) + λ Lvf
        for (int i = 0; i < nW; ++i) T.emplace_back(i, i, Mvf[i]);
        for (int k = 0; k < Lvf.outerSize(); ++k) {
            for (SpMat::InnerIterator it(Lvf, k); it; ++it) {
                T.emplace_back((int)it.row(), (int)it.col(), lambda * it.value());
            }
        }
        // C and Cᵀ
        for (int k = 0; k < C.outerSize(); ++k) {
            for (SpMat::InnerIterator it(C, k); it; ++it) {
                T.emplace_back(nW + (int)it.row(), (int)it.col(), it.value());
                T.emplace_back((int)it.col(), nW + (int)it.row(), it.value());
            }
        }
        SpMat K(nW + E, nW + E);
        K.setFromTriplets(T.begin(), T.end());
        return K;
    };

    Vec w = w0;

    auto curlNorm = [&](const Vec& ww) -> double {
        Vec r = C * ww;
        return r.norm();
    };
    double initialCurl = curlNorm(w);

    std::vector<double> lambdaHist;
    double lambda = lambdaInit;
    int totalIters = 0;

    // λ-sharpening loop: converge at one λ, then halve and continue.
    while (true) {
        lambdaHist.push_back(lambda);

        SpMat K = buildK(lambda);

        // Construct RHS once per λ and refactor.
        //   rhs_top    = − λ Lvf ŵ
        //   rhs_bottom = − C ŵ
        //
        // Since A and C are fixed at this λ, we refactor K once and
        // only rebuild the RHS each inner iteration.
        Vec rhs(nW + E);
        rhs.setZero();

        // Apply handles by zeroing K rows/cols.
        applyHandles(K, rhs, handleFaces, nW);
        K.makeCompressed();

        Eigen::SparseLU<SpMat> solver;
        solver.analyzePattern(K);
        solver.factorize(K);
        if (solver.info() != Eigen::Success) {
            // Matrix could not be factored — try a tiny regularization.
            SpMat I(K.rows(), K.cols());
            std::vector<Trip> tI;
            tI.reserve(K.rows());
            for (int i = 0; i < K.rows(); ++i) tI.emplace_back(i, i, 1e-10);
            I.setFromTriplets(tI.begin(), tI.end());
            K = K + I;
            K.makeCompressed();
            solver.analyzePattern(K);
            solver.factorize(K);
        }

        int innerIter = 0;
        for (; innerIter < maxIter; ++innerIter) {
            // Build RHS.
            rhs.head(nW) = -lambda * (Lvf * w);
            rhs.tail(E)  = -(C * w);
            // Zero RHS on handle DOFs.
            for (int f : handleFaces) {
                rhs[2*f]     = 0;
                rhs[2*f + 1] = 0;
            }

            Vec x = solver.solve(rhs);
            if (solver.info() != Eigen::Success) break;
            Vec delta = x.head(nW);

            // Step B: per-face normalization.
            double maxUpd = 0.0;
            for (int f = 0; f < F; ++f) {
                double nx = w[2*f]     + delta[2*f];
                double ny = w[2*f + 1] + delta[2*f + 1];
                double L  = std::sqrt(nx*nx + ny*ny);
                double inv = L > 1e-12 ? 1.0 / L : 0.0;
                double wxOld = w[2*f], wyOld = w[2*f+1];
                double wxNew = nx * inv;
                double wyNew = ny * inv;
                w[2*f]     = wxNew;
                w[2*f + 1] = wyNew;
                double dx = wxNew - wxOld, dy = wyNew - wyOld;
                double d2 = dx*dx + dy*dy;
                if (d2 > maxUpd) maxUpd = d2;
            }
            totalIters++;

            if (std::sqrt(maxUpd) < tol) { innerIter++; break; }
        }

        if (lambda <= lambdaMin) break;
        lambda *= 0.5;
        if (lambda < lambdaMin) lambda = lambdaMin;
    }

    Alg1Result R;
    R.w = w;
    R.initialCurl = initialCurl;
    R.finalCurl   = curlNorm(w);
    R.iterations  = totalIters;
    R.lambdaHistory = std::move(lambdaHist);
    return R;
}

} // namespace wgf
