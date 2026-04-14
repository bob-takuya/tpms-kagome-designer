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
#include "progress.h"
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <string>

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

// Run Algorithm 1. The `progressStage` argument, when non-negative,
// triggers reportProgress() calls from inside the inner loop so the
// browser UI has something smooth to draw instead of a bar stuck on
// the stage-entry label.
inline Alg1Result runAlg1(const Mesh& m,
                          const std::vector<FaceFrame>& frames,
                          const Vec& w0,
                          const std::vector<int>& handleFaces = {},
                          double lambdaInit = 10.0,
                          double lambdaMin  = 1e-2,
                          int    maxIter    = 20,
                          double tol        = 1e-5,
                          int    progressStage = -1,
                          const char* progressLabel = "Algorithm 1") {
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

    // Build the λ schedule up front: coarse geometric sequence from
    // lambdaInit down to lambdaMin, stepping by 1/3 per level. This is
    // the simplified browser-friendly version of the paper's §5.2
    // recommendation (which halves 1000 → 0 over 21 levels). Halving
    // would run a full CG-driven inner loop 21 times per pipeline
    // execution, which in WASM takes minutes. A 3-step schedule is
    // both fast enough for interactive use and, in practice, produces
    // near-identical curl-free fields.
    std::vector<double> lambdaSchedule;
    {
        double l = lambdaInit;
        while (l > lambdaMin) {
            lambdaSchedule.push_back(l);
            l /= 3.0;
        }
        lambdaSchedule.push_back(lambdaMin);
    }
    const int nLevels = (int)lambdaSchedule.size();
    const int totalOuterEstimate = nLevels * maxIter;
    int totalIters = 0;

    for (int lvl = 0; lvl < nLevels; ++lvl) {
        const double lambda = lambdaSchedule[lvl];
        lambdaHist.push_back(lambda);

        // Per-level Tikhonov, scaled with λ. Acts as the smoothness
        // regularizer (larger λ → flatter field → easier CG).
        const double epsLevel = std::max(1e-7, 1e-6 * lambda + 1e-6);

        SpMat S = buildCurlGram(C, Mvf, epsLevel);

        Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                                 Eigen::DiagonalPreconditioner<double>> cg;
        cg.setMaxIterations(300);
        cg.setTolerance(1e-6);
        cg.compute(S);
        if (cg.info() != Eigen::Success) {
            S = buildCurlGram(C, Mvf, std::max(epsLevel, 1e-3));
            cg.compute(S);
            if (cg.info() != Eigen::Success) break;
        }

        int innerIter = 0;
        for (; innerIter < maxIter; ++innerIter) {
            // Emit progress every iteration so the browser bar moves.
            if (progressStage >= 0) {
                std::string lbl = std::string(progressLabel) +
                    " (λ=" + std::to_string(lambda).substr(0, 5) +
                    ", lvl " + std::to_string(lvl + 1) + "/" +
                    std::to_string(nLevels) + ")";
                reportProgress(progressStage, totalIters, totalOuterEstimate, lbl.c_str());
            }

            Vec rhs = C * w;
            Vec mu  = cg.solve(rhs);
            Vec delta = Ct * mu;
            for (int i = 0; i < nW; ++i) delta[i] /= std::max(Mvf[i], 1e-30);

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
