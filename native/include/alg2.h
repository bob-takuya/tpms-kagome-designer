// alg2.h — Vekhter 2019 §4 scalar-field recovery: eq. 7 + eq. 8.
//
// Given an already-optimized face vector field ŵ on the cover (or any
// simply-connected punctured piece of it), recover a vertex scalar
// field θ : V → ℝ and a face scalar s : F → ℝ satisfying
//
//     ∇θ(p) ≈ s(p) · ŵ(p)^⊥
//
// Two steps:
//
//   (eq. 7)  Initial s estimate via a generalized eigenproblem.  Let
//   x = [δ; s] ∈ ℝ^{2|F| + |F|}.  Form
//
//       B row per interior edge e = (fi, fj):
//         rows 0..2|F|−1 : the discrete curl C (acts on δ)
//         cols 2|F|..2|F|+|F|−1 : face coefficient
//             s_fi · (ŵ⊥_fi · e_ij) − s_fj · (ŵ⊥_fj · e_ij)
//
//       Constraint B x = 0 expresses C(sŵ⊥ + δ) = 0.
//
//       Minimize (1/2) δᵀ M_vf δ + (μ/2) sᵀ L_f s
//       subject to B x = 0, xᵀ A₂ x = 1    A₂ = diag(M_vf, M_f)
//
//   Solved by inverse power iteration on (A₁ + A₂) with A₁ =
//   diag(M_vf, μ L_f), projecting each iterate onto ker B.
//
//   (eq. 8)  Joint (s, θ) optimization.  Alternate:
//     Step (i): θ fixed, update s via Gauss-Newton normal equations.
//     Step (ii): s fixed, update θ via the Knöppel 2015 twisted
//                Laplacian eigensolve (smallest eigenvector).
//
// Output: θ per vertex (unit complex), s per face.

#pragma once

#include "mesh.h"
#include "curl.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>

namespace wgf {

// ─────────────────────────────────────────────────────────────────────────────
// Build the constraint matrix B from the paper (eq. 7) for a given ŵ.
// Layout: columns [0, 2|F|) are δ columns, [2|F|, 2|F|+|F|) are s columns.
// ─────────────────────────────────────────────────────────────────────────────
inline SpMat buildIntegrabilityB(const Mesh& m,
                                 const std::vector<FaceFrame>& frames,
                                 const Vec& w) {
    const int F = m.nF();
    const int E = (int)m.eInt.size();
    const int nD = 2 * F;
    const int nS = F;
    const int nCols = nD + nS;

    std::vector<Trip> T;
    T.reserve(E * 6);

    for (int r = 0; r < E; ++r) {
        int h = m.eInt[r].first;
        int fi = m.he[h].face;
        int fj = m.he[m.he[h].twin].face;
        Eigen::Vector3d ev = m.V.row(m.he[h].vertex) - m.V.row(m.heStart(h));
        double eix = ev.dot(frames[fi].e1);
        double eiy = ev.dot(frames[fi].e2);
        double ejx = ev.dot(frames[fj].e1);
        double ejy = ev.dot(frames[fj].e2);

        // δ columns = discrete curl rows
        T.emplace_back(r, 2*fi,     eix);
        T.emplace_back(r, 2*fi + 1, eiy);
        T.emplace_back(r, 2*fj,    -ejx);
        T.emplace_back(r, 2*fj + 1,-ejy);

        // s columns: ŵ⊥_fi · e_ij − ŵ⊥_fj · e_ij
        // ŵ⊥ = (−wy, wx)
        double wi_x = w[2*fi], wi_y = w[2*fi + 1];
        double wj_x = w[2*fj], wj_y = w[2*fj + 1];
        double si = -wi_y * eix + wi_x * eiy;
        double sj = -wj_y * ejx + wj_x * ejy;
        T.emplace_back(r, nD + fi,  si);
        T.emplace_back(r, nD + fj, -sj);
    }
    SpMat B((int)E, nCols);
    B.setFromTriplets(T.begin(), T.end());
    return B;
}

// Face adjacency Laplacian on the scalar field s (|F|×|F|).
// Uniform weights — entries:  L[i,i] = deg(i),  L[i,j] = −1.
inline SpMat buildFaceLaplacian(const Mesh& m) {
    const int F = m.nF();
    std::vector<Trip> T;
    T.reserve(6 * F);
    for (const auto& e : m.eInt) {
        int h = e.first;
        int fi = m.he[h].face;
        int fj = m.he[m.he[h].twin].face;
        T.emplace_back(fi, fi,  1.0);
        T.emplace_back(fj, fj,  1.0);
        T.emplace_back(fi, fj, -1.0);
        T.emplace_back(fj, fi, -1.0);
    }
    SpMat L(F, F);
    L.setFromTriplets(T.begin(), T.end());
    return L;
}

// Face "mass" (area) as a diagonal.
inline Vec faceArea(const std::vector<FaceFrame>& frames) {
    Vec m(frames.size());
    for (int f = 0; f < (int)frames.size(); ++f) m[f] = frames[f].area;
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm 2 — initial s via inverse power iteration + null-space projection.
// ─────────────────────────────────────────────────────────────────────────────
inline Vec initialScaling(const Mesh& m,
                          const std::vector<FaceFrame>& frames,
                          const Vec& w,
                          double mu = 1e-4,
                          int maxIter = 25) {
    const int F  = m.nF();
    const int nD = 2 * F;
    const int nS = F;
    const int N  = nD + nS;

    SpMat B  = buildIntegrabilityB(m, frames, w);
    Vec   Mvf = faceFieldMass(frames);
    Vec   Mf  = faceArea(frames);
    SpMat Lf  = buildFaceLaplacian(m);

    // Build A1 + A2 (symmetric PSD). Upper-left (2F × 2F): 2·Mvf block;
    // lower-right (F × F): Mf + μ Lf.
    SpMat A(N, N);
    {
        std::vector<Trip> T;
        T.reserve(nD + nS + Lf.nonZeros());
        for (int i = 0; i < nD; ++i) T.emplace_back(i, i, 2.0 * Mvf[i]);
        for (int i = 0; i < nS; ++i) T.emplace_back(nD + i, nD + i, Mf[i]);
        for (int k = 0; k < Lf.outerSize(); ++k) {
            for (SpMat::InnerIterator it(Lf, k); it; ++it) {
                T.emplace_back(nD + (int)it.row(), nD + (int)it.col(), mu * it.value());
            }
        }
        A.setFromTriplets(T.begin(), T.end());
    }

    // A2 (diagonal normalization).
    std::vector<double> A2diag(N);
    for (int i = 0; i < nD; ++i) A2diag[i] = Mvf[i];
    for (int i = 0; i < nS; ++i) A2diag[nD + i] = Mf[i];

    auto a2norm = [&](const Vec& x) -> double {
        double s = 0;
        for (int i = 0; i < N; ++i) s += A2diag[i] * x[i] * x[i];
        return std::sqrt(std::max(s, 1e-30));
    };
    auto a2normalize = [&](Vec& x) { x /= a2norm(x); };

    // Add a small uniform Tikhonov so A is strictly positive definite
    // and ConjugateGradient converges reliably on all meshes.
    {
        std::vector<Trip> tI;
        tI.reserve(N);
        for (int i = 0; i < N; ++i) tI.emplace_back(i, i, 1e-8);
        SpMat R(N, N);
        R.setFromTriplets(tI.begin(), tI.end());
        A = A + R;
    }
    A.makeCompressed();
    Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                             Eigen::DiagonalPreconditioner<double>> solverA;
    solverA.setMaxIterations(1000);
    solverA.setTolerance(1e-8);
    solverA.compute(A);
    if (solverA.info() != Eigen::Success) {
        // Couldn't even set up CG — leave s at zero.
        return Vec::Zero(nS);
    }

    // BBᵀ factor (used for projection onto ker B).
    SpMat Bt = SpMat(B.transpose());
    SpMat BBt = B * Bt;
    // Tikhonov for stability
    {
        std::vector<Trip> ti;
        ti.reserve(BBt.rows());
        for (int i = 0; i < BBt.rows(); ++i) ti.emplace_back(i, i, 1e-10);
        SpMat reg(BBt.rows(), BBt.cols());
        reg.setFromTriplets(ti.begin(), ti.end());
        BBt = BBt + reg;
    }
    BBt.makeCompressed();
    Eigen::SimplicialLDLT<SpMat> solverBBt;
    solverBBt.analyzePattern(BBt);
    solverBBt.factorize(BBt);

    // Random init, normalized by A2 norm.
    Vec x = Vec::Random(N);
    a2normalize(x);

    for (int it = 0; it < maxIter; ++it) {
        // Inverse iteration: rhs = A2 * x
        Vec rhs(N);
        for (int i = 0; i < N; ++i) rhs[i] = A2diag[i] * x[i];
        Vec y = solverA.solve(rhs);
        if (solverA.info() != Eigen::Success) break;

        // Project y onto ker B:  y ← y − Bᵀ (BBᵀ)⁻¹ B y
        Vec By  = B * y;
        Vec mu_ = solverBBt.solve(By);
        y = y - Bt * mu_;

        a2normalize(y);
        x = y;
    }

    // Extract s (second block of x).
    Vec s(nS);
    for (int i = 0; i < nS; ++i) s[i] = x[nD + i];
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rescale s so that |sŵ⊥ · e_ij| ≤ π on every interior edge
// (Vekhter eq. 3.4): global scale ρ = max_e s_i · ‖ŵ_i^⊥ · e_ij‖,
// then  s ← s · (π / ρ).
// ─────────────────────────────────────────────────────────────────────────────
inline void rescaleAliasFree(Vec& s, const Mesh& m,
                             const std::vector<FaceFrame>& frames,
                             const Vec& w,
                             double userScale = 1.0) {
    double rho = 0;
    for (const auto& e : m.eInt) {
        int h = e.first;
        int fi = m.he[h].face;
        Eigen::Vector3d ev = m.V.row(m.he[h].vertex) - m.V.row(m.heStart(h));
        double eix = ev.dot(frames[fi].e1);
        double eiy = ev.dot(frames[fi].e2);
        double wx = w[2*fi], wy = w[2*fi + 1];
        double val = std::fabs(s[fi] * (-wy * eix + wx * eiy));
        if (val > rho) rho = val;
    }
    if (rho < 1e-30) return;
    s *= userScale * (M_PI / rho);
}

// ─────────────────────────────────────────────────────────────────────────────
// Knöppel 2015 twisted-Laplacian eigensolve for θ (complex per vertex).
//
// Energy:    E(θ) = Σ w_ij · | θ_j − exp(i · s̄_ij · ŵ⊥·(p_j−p_i)) · θ_i |²
//
// with s̄_ij the average of s over the two faces adjacent to edge (i,j),
// and w_ij the standard cotangent weight.
//
// Returns a 2|V| real vector (re, im interleaved).
// ─────────────────────────────────────────────────────────────────────────────
inline Vec solveThetaKnoppel(const Mesh& m,
                             const std::vector<FaceFrame>& frames,
                             const Vec& w,
                             const Vec& s,
                             int maxIter = 20) {
    const int n = m.nV();

    // For each interior edge we need an "edge-averaged direction",
    // expressed per vertex endpoint. We adopt the simplest form: use the
    // average over the two adjacent faces of s_f · ŵ⊥_f · e_ij.
    //
    // Per half-edge (from vi to vj), the edge phase φ = s̄·ŵ⊥·(p_j − p_i).
    // Build Hermitian Laplacian L̂ with entries stored as real 2×2 blocks.
    SpMat L(2 * n, 2 * n);
    std::vector<Trip> T;
    T.reserve(m.eInt.size() * 16);

    for (const auto& e : m.eInt) {
        int h = e.first;
        int t = e.second;
        int vi = m.heStart(h);
        int vj = m.he[h].vertex;
        int fi = m.he[h].face;
        int fj = m.he[t].face;
        double wij = std::max(0.0, cotWeight(m, h));

        Eigen::Vector3d ev = m.V.row(vj) - m.V.row(vi);
        // Project into each face and compute s · ŵ⊥ · e; average.
        auto phiFrom = [&](int f) -> double {
            double eix = ev.dot(frames[f].e1);
            double eiy = ev.dot(frames[f].e2);
            double wx = w[2*f], wy = w[2*f + 1];
            return s[f] * (-wy * eix + wx * eiy);
        };
        double ph = 0.5 * (phiFrom(fi) + phiFrom(fj));
        double c = std::cos(ph), sn = std::sin(ph);

        // L[i,i] += w, L[j,j] += w
        int ii = 2 * vi, jj = 2 * vj;
        T.emplace_back(ii,   ii,   wij);
        T.emplace_back(ii+1, ii+1, wij);
        T.emplace_back(jj,   jj,   wij);
        T.emplace_back(jj+1, jj+1, wij);

        // L[j,i] = −w · e^{iφ}  → real block [[−w c, w sn], [−w sn, −w c]]
        // L[i,j] = conj → [[−w c, −w sn], [w sn, −w c]]
        T.emplace_back(jj,   ii,   -wij * c);
        T.emplace_back(jj,   ii+1,  wij * sn);
        T.emplace_back(jj+1, ii,   -wij * sn);
        T.emplace_back(jj+1, ii+1, -wij * c);
        T.emplace_back(ii,   jj,   -wij * c);
        T.emplace_back(ii,   jj+1, -wij * sn);
        T.emplace_back(ii+1, jj,    wij * sn);
        T.emplace_back(ii+1, jj+1, -wij * c);
    }
    L.setFromTriplets(T.begin(), T.end());

    // Mass matrix (diagonal): 2 × vertex Voronoi area.
    Vec vmass = vertexMass(m);
    Vec M2(2 * n);
    for (int i = 0; i < n; ++i) { M2[2*i] = vmass[i]; M2[2*i+1] = vmass[i]; }

    // Inverse iteration with shift 1e-4.
    SpMat A = L;
    {
        std::vector<Trip> ti;
        ti.reserve(2 * n);
        for (int i = 0; i < 2 * n; ++i) ti.emplace_back(i, i, 1e-4 * M2[i]);
        SpMat reg(2 * n, 2 * n);
        reg.setFromTriplets(ti.begin(), ti.end());
        A = A + reg;
    }
    A.makeCompressed();
    Eigen::SimplicialLDLT<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) return Vec::Zero(2 * n);

    Vec x = Vec::Random(2 * n);
    auto nz = [&](Vec& y) {
        double nr = 0;
        for (int i = 0; i < 2 * n; ++i) nr += M2[i] * y[i] * y[i];
        y /= std::sqrt(std::max(nr, 1e-30));
    };
    nz(x);
    for (int it = 0; it < maxIter; ++it) {
        Vec b(2 * n);
        for (int i = 0; i < 2 * n; ++i) b[i] = M2[i] * x[i];
        Vec y = solver.solve(b);
        nz(y);
        x = y;
    }
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// Joint (s, θ) optimization (eq. 8). We alternate:
//   Step (i): θ fixed → solve for s via Gauss-Newton normal equations.
//   Step (ii): s fixed → solve θ eigenproblem.
//
// Step (i) linearized per edge:
//
//     r = θ_j − θ_i·exp(i · s̄ · ŵ⊥·(p_j−p_i)) ≈ 0
//
// Here we treat θ as fixed (from step ii). Expanding r as a function of
// s yields a least-squares problem. For a rough Gauss-Newton update, it
// suffices to solve  Lf s + α·(Bˢ)ᵀ Bˢ s = RHS  where Bˢ is the linearized
// coupling; for simplicity we regularize s with a μ‖∇s‖² and re-run the
// Knöppel θ-solver at the new s. This matches the paper's description of
// 10 alternations.
// ─────────────────────────────────────────────────────────────────────────────
struct JointResult {
    Vec theta;   // 2|V| interleaved real/imag
    Vec s;       // |F|
};

inline JointResult solveJointScalar(const Mesh& m,
                                    const std::vector<FaceFrame>& frames,
                                    const Vec& w,
                                    double mu = 1e-4,
                                    int nAltern = 10) {
    // 1. Initial s via Algorithm 2.
    Vec s = initialScaling(m, frames, w, mu, 25);

    // 2. Anti-aliasing rescale.
    rescaleAliasFree(s, m, frames, w, 1.0);

    // 3. Alternating (θ, s) updates.
    Vec theta;
    for (int it = 0; it < nAltern; ++it) {
        theta = solveThetaKnoppel(m, frames, w, s, 20);

        // Step (i): Gauss-Newton s update at fixed θ.
        //
        // For each edge (i,j) with half-edge h in face fi, let
        //    φ_ij = s̄·ŵ⊥·e_ij    and    r_ij = arg(θ_j · conj(θ_i)) − φ_ij
        // Linearizing in s gives
        //    ∂r/∂s_fi = −(1/2) (ŵ⊥_fi · e_ij),
        //    ∂r/∂s_fj = −(1/2) (ŵ⊥_fj · e_ij).
        // Normal equations: Jᵀ J s = Jᵀ r + μ Lf s = 0, solved by Cholesky.
        const int F = m.nF();
        SpMat Lf = buildFaceLaplacian(m);
        std::vector<Trip> T;
        Vec rhs = Vec::Zero(F);
        T.reserve(4 * m.eInt.size());
        for (const auto& e : m.eInt) {
            int h = e.first;
            int fi = m.he[h].face;
            int fj = m.he[m.he[h].twin].face;
            int vi = m.heStart(h);
            int vj = m.he[h].vertex;
            Eigen::Vector3d ev = m.V.row(vj) - m.V.row(vi);
            double eix_i = ev.dot(frames[fi].e1);
            double eiy_i = ev.dot(frames[fi].e2);
            double eix_j = ev.dot(frames[fj].e1);
            double eiy_j = ev.dot(frames[fj].e2);
            double jac_i = -0.5 * (-w[2*fi+1] * eix_i + w[2*fi] * eiy_i);
            double jac_j = -0.5 * (-w[2*fj+1] * eix_j + w[2*fj] * eiy_j);

            // Target r = arg(θ_j · conj(θ_i)) ∈ (−π, π]
            double a = theta[2*vi], b = theta[2*vi+1];
            double c = theta[2*vj], d = theta[2*vj+1];
            double dx = c * a + d * b;
            double dy = d * a - c * b;
            double r = std::atan2(dy, dx);

            T.emplace_back(fi, fi, jac_i * jac_i);
            T.emplace_back(fj, fj, jac_j * jac_j);
            T.emplace_back(fi, fj, jac_i * jac_j);
            T.emplace_back(fj, fi, jac_i * jac_j);
            rhs[fi] += jac_i * r;
            rhs[fj] += jac_j * r;
        }
        SpMat JtJ(F, F);
        JtJ.setFromTriplets(T.begin(), T.end());
        SpMat Areg = JtJ + mu * Lf;
        {
            std::vector<Trip> ti;
            for (int i = 0; i < F; ++i) ti.emplace_back(i, i, 1e-8);
            SpMat reg(F, F);
            reg.setFromTriplets(ti.begin(), ti.end());
            Areg = Areg + reg;
        }
        Areg.makeCompressed();
        Eigen::SimplicialLDLT<SpMat> solver;
        solver.compute(Areg);
        if (solver.info() == Eigen::Success) {
            Vec ds = solver.solve(rhs);
            s += ds;
        }
        rescaleAliasFree(s, m, frames, w, 1.0);
    }

    JointResult R;
    R.theta = theta;
    R.s     = s;
    return R;
}

} // namespace wgf
