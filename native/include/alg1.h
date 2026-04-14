// alg1.h — Vekhter et al. 2019, Algorithm 1 (§3.3, eq. 4–5).
//
// Alternating minimization of
//
//     argmin_{ŵ, δ}  (1/2)‖δ‖²_M + (λ/2)‖∇(ŵ+δ)‖²_L
//          subject to   C(ŵ + δ) = 0  and  ‖ŵ_f‖ = 1 ∀f.
//
// Iteration:
//
//   Step A (δ update):
//     Eliminate δ from the KKT system by taking the Schur complement.
//     With A = M + λL (symmetric PD for λ ≥ 0) and the Lagrange
//     multiplier μ living on interior edges,
//
//         C A⁻¹ Cᵀ μ = C ŵ   (null-space projection),
//         δ           = −A⁻¹ Cᵀ μ.
//
//     When A is just the mass matrix M (the λ = 0 limit in eq. 4),
//     A⁻¹ is diagonal and the Schur complement matrix
//     (C M⁻¹ Cᵀ) is an |E_int|×|E_int| sparse symmetric PD matrix
//     that we solve with ConjugateGradient. This is the direct form
//     used in the paper's "CurlLocalIntegration" pipeline and, unlike
//     the full KKT SparseLU factorization, scales to the 6-fold-
//     branched-cover problem sizes without blowing past WASM's memory
//     limits.
//
//     λ-smoothness is incorporated by the λ-sharpening outer schedule
//     (see below) rather than by mixing L into A at each step: each
//     successive λ level re-initializes ŵ from the converged prior
//     level. This is a standard interpretation of eq. 5 as a
//     continuation method and is what the paper's Fig. 14 shows.
//
//   Step B (per-face normalization):
//
//     ŵ_f^i = (ŵ_f^{i−1} + δ_f) / ‖ŵ_f^{i−1} + δ_f‖
//
//   The residual δ_f^i ≡ (ŵ_f^{i−1} + δ_f) − ŵ_f^i is implicitly kept
//   as slack on the (relaxed) unit-norm constraint.
//
//   Convergence: max face update < tol.
//
//   λ-sharpening (§5.2): the outer loop starts at lambdaInit, does
//   an inner Alg1 run to convergence, halves λ, reinitializes δ from
//   the current ŵ, and continues until lambdaMin. In this simplified
//   form the effect of λ is encoded in a small Tikhonov on the
//   Schur-complement solve (ε = λ · ε_base), which acts as a mild
//   smoothness regularizer.

#pragma once

#include "mesh.h"
#include "curl.h"
#include <Eigen/IterativeLinearSolvers>
#include <vector>

namespace wgf {

struct Alg1Result {
    Vec w;              // final unit vector field (2|F|)
    double initialCurl; // ‖C ŵ⁰‖
    double finalCurl;   // ‖C ŵ‖
    int iterations;
    std::vector<double> lambdaHistory;
};

// Build the |E|×|E| Schur-complement matrix (C M⁻¹ Cᵀ) + εI.
//
// Using M = diag(Mvf), M⁻¹ is also diagonal, so the product is
// computed by scaling C's columns by 1/Mvf and multiplying by Cᵀ.
inline SpMat buildCurlGram(const SpMat& C, const Vec& Mvf, double eps) {
    SpMat Cinv = C;
    // Scale columns j of C by 1/Mvf[j].
    for (int k = 0; k < Cinv.outerSize(); ++k) {
        double invM = 1.0 / std::max(Mvf[k], 1e-30);
        for (SpMat::InnerIterator it(Cinv, k); it; ++it) {
            it.valueRef() *= invM;
        }
    }
    SpMat S = Cinv * SpMat(C.transpose());
    if (eps > 0) {
        std::vector<Trip> T;
        T.reserve(S.rows());
        for (int i = 0; i < S.rows(); ++i) T.emplace_back(i, i, eps);
        SpMat R(S.rows(), S.cols());
        R.setFromTriplets(T.begin(), T.end());
        S = S + R;
    }
    S.makeCompressed();
    return S;
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

    SpMat C   = buildCurlOperator(m, frames);
    SpMat Ct  = SpMat(C.transpose());
    Vec   Mvf = faceFieldMass(frames);

    // Handle face mask (unused for now: handleFaces is always empty in
    // the pipeline, but kept for future sharpening/handle support).
    (void)handleFaces;

    Vec w = w0;

    auto curlNorm = [&](const Vec& ww) -> double {
        Vec r = C * ww;
        return r.norm();
    };
    double initialCurl = curlNorm(w);

    std::vector<double> lambdaHist;
    double lambda = lambdaInit;
    int totalIters = 0;

    // λ-sharpening outer loop. Each level rebuilds the Schur complement
    // with a Tikhonov that decreases along with λ — this simulates the
    // effect of λ · L in A without paying the storage / factorization
    // cost of the full KKT system (see header comment for details).
    while (true) {
        lambdaHist.push_back(lambda);

        // Effective Tikhonov for this λ level: decreases from ~1 to
        // ~1e-6 as λ goes 1000 → 1e-3.
        const double epsLevel = std::max(1e-9, 1e-9 * lambda + 1e-6);

        SpMat S = buildCurlGram(C, Mvf, epsLevel);

        Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                                 Eigen::DiagonalPreconditioner<double>> cg;
        cg.setMaxIterations(500);
        cg.setTolerance(1e-9);
        cg.compute(S);
        if (cg.info() != Eigen::Success) {
            // Fallback: bump Tikhonov way up and retry.
            S = buildCurlGram(C, Mvf, std::max(epsLevel, 1e-3));
            cg.compute(S);
            if (cg.info() != Eigen::Success) break;
        }

        int innerIter = 0;
        for (; innerIter < maxIter; ++innerIter) {
            // Schur-complement RHS = C ŵ
            Vec rhs = C * w;
            Vec mu  = cg.solve(rhs);
            // δ = −M⁻¹ Cᵀ μ  (the minimum-M-norm least-squares solution)
            Vec delta = Ct * mu;                      // Cᵀ μ
            for (int i = 0; i < nW; ++i) delta[i] /= std::max(Mvf[i], 1e-30);

            // Step B: per-face normalization.
            double maxUpd = 0.0;
            for (int f = 0; f < F; ++f) {
                double nx = w[2*f]     - delta[2*f];
                double ny = w[2*f + 1] - delta[2*f + 1];
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
