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
#include <Eigen/SparseLU>
#include <algorithm>
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

// Run Algorithm 1. The `progressStage` argument, when non-negative,
// triggers reportProgress() calls from inside the inner loop so the
// browser UI has something smooth to draw instead of a bar stuck on
// the stage-entry label.
inline Alg1Result runAlg1(const Mesh& m,
                          const std::vector<FaceFrame>& frames,
                          const Vec& w0,
                          const std::vector<int>& handleFaces = {},
                          double lambdaInit = 1000.0,
                          double lambdaMin  = 1e-3,
                          int    maxIter    = 50,
                          double tol        = 1e-5,
                          int    progressStage = -1,
                          const char* progressLabel = "Algorithm 1") {
    const int F = m.nF();
    const int nW = 2 * F;

    SpMat C    = buildCurlOperator(m, frames);
    Vec   Mvf  = faceFieldMass(frames);
    SpMat Lvf  = buildFaceDirichlet(m, frames);
    const int E = C.rows();

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

    // Build the λ schedule up front (paper §5.2): progressively reduce
    // λ by 1/2 to sharpen the field while preserving a robust smooth
    // warm-start from the previous level.
    std::vector<double> lambdaSchedule;
    {
        double l = lambdaInit;
        while (l > lambdaMin) {
            lambdaSchedule.push_back(l);
            l *= 0.5;
        }
        lambdaSchedule.push_back(lambdaMin);
    }
    const int nLevels = (int)lambdaSchedule.size();
    const int totalOuterEstimate = nLevels * maxIter;
    const int progressTotal = std::max(1, totalOuterEstimate);
    int totalIters = 0;

    if (progressStage >= 0) {
        std::string lbl = std::string(progressLabel) + " (initializing)";
        reportProgress(progressStage, 0, progressTotal, lbl.c_str());
    }

    // The KKT sparsity pattern is identical across λ-levels (only the
    // λ·Lvf and Mvf values vary), so SparseLU's symbolic factorization
    // can be reused. analyzePattern is ~20–50% of the cost of a single
    // factorize call, and we'd pay it nLevels times otherwise.
    Eigen::SparseLU<SpMat> lu;
    bool patternAnalyzed = false;

    for (int lvl = 0; lvl < nLevels; ++lvl) {
        const double lambda = lambdaSchedule[lvl];
        lambdaHist.push_back(lambda);

        // Factor the KKT matrix once per λ-level:
        // [M + λL, C^T;
        //  C,      0  ] [δ; μ] = [-λL u; -C u]
        std::vector<Trip> KK;
        KK.reserve((size_t)nW + Lvf.nonZeros() + C.nonZeros() * 2 + E);
        for (int i = 0; i < nW; ++i) KK.emplace_back(i, i, std::max(Mvf[i], 1e-12));
        for (int k = 0; k < Lvf.outerSize(); ++k) {
            for (SpMat::InnerIterator it(Lvf, k); it; ++it) {
                KK.emplace_back((int)it.row(), (int)it.col(), lambda * it.value());
            }
        }
        for (int k = 0; k < C.outerSize(); ++k) {
            for (SpMat::InnerIterator it(C, k); it; ++it) {
                const int r = (int)it.row();
                const int c = (int)it.col();
                const double v = it.value();
                KK.emplace_back(nW + r, c, v);
                KK.emplace_back(c, nW + r, v);
            }
        }
        // tiny regularization on μ block for numerical robustness
        for (int i = 0; i < E; ++i) KK.emplace_back(nW + i, nW + i, 1e-12);

        SpMat KKT(nW + E, nW + E);
        KKT.setFromTriplets(KK.begin(), KK.end());
        KKT.makeCompressed();

        // KKT 行列の統計を出力
        {
            double diagMin = 1e300, diagMax = -1e300, diagAbsMin = 1e300;
            int negDiag = 0, zeroDiag = 0;
            for (int i = 0; i < KKT.rows(); ++i) {
                double d = KKT.coeff(i, i);
                diagMin = std::min(diagMin, d);
                diagMax = std::max(diagMax, d);
                diagAbsMin = std::min(diagAbsMin, std::abs(d));
                if (d < 0) negDiag++;
                if (std::abs(d) < 1e-12) zeroDiag++;
            }
            fprintf(stderr,
                "[alg1-diag] lvl=%d lambda=%.4g KKT %dx%d nnz=%d "
                "diag[min=%.4g max=%.4g absMin=%.4g negCount=%d zeroCount=%d]\n",
                lvl, lambda, (int)KKT.rows(), (int)KKT.cols(),
                (int)KKT.nonZeros(),
                diagMin, diagMax, diagAbsMin, negDiag, zeroDiag);
        }

        // cotan 重みの統計（最初のレベルのみ）
        if (lvl == 0) {
            double cotMin = 1e300, cotMax = -1e300;
            int cotHuge = 0;
            for (const auto& e : m.eInt) {
                double cw = cotWeight(m, e.first);
                cotMin = std::min(cotMin, cw);
                cotMax = std::max(cotMax, cw);
                if (std::abs(cw) > 100.0) cotHuge++;
            }
            fprintf(stderr,
                "[alg1-diag] cotan weights: min=%.4g max=%.4g huge(>100)=%d / %d edges\n",
                cotMin, cotMax, cotHuge, (int)m.eInt.size());
        }

        if (progressStage >= 0) {
            std::string lbl = std::string(progressLabel)
                + " (factorizing λ=" + std::to_string(lambda).substr(0, 5)
                + " lvl " + std::to_string(lvl + 1) + "/" + std::to_string(nLevels) + ")";
            reportProgress(progressStage, totalIters, progressTotal, lbl.c_str());
        }

        if (!patternAnalyzed) {
            lu.analyzePattern(KKT);
            patternAnalyzed = true;
        }
        lu.factorize(KKT);
        if (lu.info() != Eigen::Success) {
            fprintf(stderr,
                "[alg1-diag] SparseLU FAILED lvl=%d lambda=%.4g info=%d\n",
                lvl, lambda, (int)lu.info());
            break;
        }
        fprintf(stderr, "[alg1-diag] SparseLU OK lvl=%d lambda=%.4g\n", lvl, lambda);

        if (progressStage >= 0) {
            std::string lbl = std::string(progressLabel)
                + " (solving λ=" + std::to_string(lambda).substr(0, 5)
                + " lvl " + std::to_string(lvl + 1) + "/" + std::to_string(nLevels) + ")";
            reportProgress(progressStage, totalIters, progressTotal, lbl.c_str());
        }

        int innerIter = 0;
        for (; innerIter < maxIter; ++innerIter) {
            // Emit progress every iteration so the browser bar moves.
            if (progressStage >= 0) {
                std::string lbl = std::string(progressLabel) +
                    " (λ=" + std::to_string(lambda).substr(0, 5) +
                    ", lvl " + std::to_string(lvl + 1) + "/" +
                    std::to_string(nLevels) + ")";
                reportProgress(progressStage, totalIters, progressTotal, lbl.c_str());
            }

            Vec rhs(nW + E);
            rhs.setZero();
            rhs.head(nW) = -lambda * (Lvf * w);
            rhs.tail(E)  = -(C * w);
            Vec sol = lu.solve(rhs);
            if (lu.info() != Eigen::Success) break;
            Vec delta = sol.head(nW);

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
                // Residual on the unit-norm projection step:
                // δ^i_f = (w^{i-1}_f + δ̃_f) - w^i_f
                double dx = wxOld + delta[2*f]     - wxNew;
                double dy = wyOld + delta[2*f + 1] - wyNew;
                double d2 = dx * dx + dy * dy;
                if (d2 > maxUpd) maxUpd = d2;
            }
            totalIters++;

            if (std::sqrt(maxUpd) < tol) { innerIter++; break; }
        }
    }

    if (progressStage >= 0) {
        std::string lbl = std::string(progressLabel) + " (done)";
        reportProgress(progressStage, progressTotal, progressTotal, lbl.c_str());
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
