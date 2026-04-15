// curl_local_integration.h
//
// Verbatim port of
//   CurlLocalIntegration::locallyIntegrateOneComponent
// from the13fools/weaving-geodesic-foliations, adapted to the Mesh /
// half-edge structures used here.
//
// Computes a per-face scalar s that, when applied to the face direction
// field w, makes s*w as close to curl-free as possible while staying
// smooth. Formulated as a generalized eigenproblem
//
//     min_s  s^T (D^T D + sreg * L_face) s  /  s^T M s
//
// where
//   D         is the |E_int| x |F| discrete curl matrix
//   L_face    is D_dual^T * inv(edge_metric) * D_dual
//             (a face Laplacian weighted by the cotangent edge metric)
//   M         is the diagonal face-area mass matrix
//
// The smallest generalized eigenvector (solved via inverse power
// iteration) gives s up to sign, which we then flip so that the sum
// is positive (matches the reference's sign-fix step).

#pragma once

#include "mesh.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <vector>
#include <cmath>

namespace wgf {

// Compute the face Laplacian L_face and the per-face-area mass M.
// Returns: L_face (|F|x|F|), M (diagonal, |F|x|F|).
inline void buildFaceLaplacianCot(
    const Mesh& mesh,
    const std::vector<FaceFrame>& frames,
    SpMat& Lface,
    SpMat& M) {

    const int nF = mesh.nF();

    // Build edge metric from cotangent half-weights. For each interior
    // edge with half-edge h we use cotWeight(mesh, h) — it returns the
    // sum of the two opposite cotangents. The reference uses libigl's
    // cotmatrix_entries which stores the per-face-corner cotangent and
    // accumulates across the two adjacent faces; summing cotWeight once
    // per interior edge is equivalent.
    std::vector<double> edgeMetric(mesh.eInt.size(), 0.0);
    for (size_t e = 0; e < mesh.eInt.size(); ++e) {
        int h = mesh.eInt[e].first;
        edgeMetric[e] = std::max(1e-30, cotWeight(mesh, h));
    }

    // Dface: |E_int| x |F|, with +1 at one adjacent face and -1 at the
    // other (arbitrary but consistent orientation). This is a face dual
    // "gradient" operator.
    std::vector<Trip> Dcoef;
    Dcoef.reserve(2 * mesh.eInt.size());
    for (size_t e = 0; e < mesh.eInt.size(); ++e) {
        int h = mesh.eInt[e].first;
        int t = mesh.eInt[e].second;
        int f0 = mesh.he[h].face;
        int f1 = mesh.he[t].face;
        Dcoef.emplace_back((int)e, f0, -1.0);
        Dcoef.emplace_back((int)e, f1,  1.0);
    }
    SpMat Dface((int)mesh.eInt.size(), nF);
    Dface.setFromTriplets(Dcoef.begin(), Dcoef.end());

    // inverse edge metric (diagonal, as sparse)
    std::vector<Trip> invEM;
    invEM.reserve(mesh.eInt.size());
    for (size_t e = 0; e < mesh.eInt.size(); ++e) {
        invEM.emplace_back((int)e, (int)e, 1.0 / edgeMetric[e]);
    }
    SpMat invEdgeMetric((int)mesh.eInt.size(), (int)mesh.eInt.size());
    invEdgeMetric.setFromTriplets(invEM.begin(), invEM.end());

    Lface = SpMat(Dface.transpose()) * invEdgeMetric * Dface;

    // Face-area mass
    std::vector<Trip> Mtri;
    Mtri.reserve(nF);
    for (int f = 0; f < nF; ++f) {
        Mtri.emplace_back(f, f, frames[f].area);
    }
    M = SpMat(nF, nF);
    M.setFromTriplets(Mtri.begin(), Mtri.end());
}

// Compute per-face s from the direction field v.
inline void curlLocalIntegration(
    const Mesh& mesh,
    const std::vector<FaceFrame>& frames,
    const Vec& v,      // 2|F| face direction vectors
    double sreg,
    Vec& s) {

    const int nF = mesh.nF();

    SpMat Lface, M;
    buildFaceLaplacianCot(mesh, frames, Lface, M);

    // Discrete curl D: one row per interior edge. For an interior edge
    // shared by faces (f0, f1), with vertices (v0, v1) and edge vector
    // e_vec = V[v1] - V[v0], the curl coefficient at face f is
    //     ( ( w_f x n_f ) . e_vec )
    // (i.e. (w^perp) . e). Matches the reference's use of Bs * Js * field.
    const int nE = (int)mesh.eInt.size();
    std::vector<Trip> Dcoef;
    Dcoef.reserve(2 * nE);
    int row = 0;
    for (int i = 0; i < nE; ++i) {
        int h = mesh.eInt[i].first;
        int t = mesh.eInt[i].second;
        int f0 = mesh.he[h].face;
        int f1 = mesh.he[t].face;
        int va = mesh.heStart(h);
        int vb = mesh.he[h].vertex;
        Eigen::Vector3d ev = mesh.V.row(vb) - mesh.V.row(va);

        auto wPerpDotEdge = [&](int f) -> double {
            double vx = v[2*f], vy = v[2*f + 1];
            Eigen::Vector3d fw = vx * frames[f].e1 + vy * frames[f].e2;
            fw = fw.cross(frames[f].n);
            double nr = fw.norm();
            if (nr < 1e-30) return 0.0;
            return (fw / nr).dot(ev);
        };

        Dcoef.emplace_back(row, f0, -wPerpDotEdge(f0));
        Dcoef.emplace_back(row, f1,  wPerpDotEdge(f1));
        row++;
    }
    SpMat D(row, nF);
    D.setFromTriplets(Dcoef.begin(), Dcoef.end());

    SpMat op = SpMat(D.transpose()) * D + sreg * Lface;

    // Add a tiny ridge so LDLT succeeds even on rank-deficient ops.
    {
        std::vector<Trip> rtri;
        rtri.reserve(nF);
        for (int i = 0; i < nF; ++i) rtri.emplace_back(i, i, 1e-12);
        SpMat R(nF, nF);
        R.setFromTriplets(rtri.begin(), rtri.end());
        op += R;
    }
    op.makeCompressed();

    Eigen::SimplicialLDLT<SpMat> solver;
    solver.compute(op);
    if (solver.info() != Eigen::Success) {
        s = Vec::Ones(nF);
        return;
    }

    // Inverse power iteration with M as the weight (generalized eigenproblem).
    Vec x = Vec::Random(nF);
    auto mnorm = [&](const Vec& vv) {
        double acc = 0;
        for (int i = 0; i < nF; ++i) acc += vv[i] * frames[i].area * vv[i];
        return std::sqrt(std::max(acc, 1e-30));
    };
    x /= mnorm(x);
    for (int i = 0; i < 1000; ++i) {
        Vec rhs(nF);
        for (int k = 0; k < nF; ++k) rhs[k] = frames[k].area * x[k];
        Vec nx = solver.solve(rhs);
        double nn = mnorm(nx);
        x = nx / nn;
    }

    s = x;
    // Sign fix: the reference flips so that total-s is positive.
    double total = s.sum();
    if (total < 0) s = -s;
}

} // namespace wgf
