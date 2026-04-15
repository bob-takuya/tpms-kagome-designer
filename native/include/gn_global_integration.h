// gn_global_integration.h
//
// Verbatim port of
//   GNGlobalIntegration::globallyIntegrateOneComponent
// from the13fools/weaving-geodesic-foliations.
//
// Alternates between:
//   (theta step) Inverse power iteration on a real 2n x 2n "connection
//                Laplacian" whose off-diagonal 2x2 blocks are rotation
//                matrices R(s * w^perp . edge). The smallest eigenvector
//                encodes theta at each vertex as a (cos, sin) pair, and
//                we recover theta via atan2.
//
//   (s step)    For each face, solve a 1x1 normal equation that picks
//                the scale minimizing Sum_e (theta_diff - s * unscaled_diff)^2
//                along the face's 3 edges, with theta_diff wrapped into
//                (-pi, pi].
//
// Inputs:
//   mesh     base triangle mesh
//   v        per-face 2D direction vector (in face-local frame)
//   scales   per-face scalar (will be read AND modified)
//   outerIters  number of alternations (paper uses ~6)
//   powerIters  inverse-power iterations per theta step (paper uses ~20)
//
// Output:
//   theta    per-vertex real angular scalar in (-pi, pi]
//   scales   updated in place

#pragma once

#include "mesh.h"
#include "progress.h"
#include <Eigen/SparseCholesky>
#include <vector>
#include <cstdio>

namespace wgf {

inline void gnGlobalIntegration(
    const Mesh& mesh,
    const std::vector<FaceFrame>& frames,
    const Vec& v,                          // 2 * nF face direction vectors
    Vec& scales,                           // nF, in/out
    Vec& theta,                            // nV, out
    int outerIters = 6,
    int powerIters = 20,
    int progressStage = -1,
    const char* progressLabel = "GN global integration") {

    const int nF = mesh.nF();
    const int nV = mesh.nV();

    theta = Vec::Zero(nV);

    // Precompute the 3 "rowsL / colsL / difVecUnscaled" triples per face.
    // rowsL[3f+e] and colsL[3f+e] are the edge endpoints (ordered per
    // reference code). difVecUnscaled[3f+e] is the dot product of the
    // face's w^perp with that directed edge.
    std::vector<int>    rowsL;   rowsL.reserve((size_t)3 * nF);
    std::vector<int>    colsL;   colsL.reserve((size_t)3 * nF);
    std::vector<double> difVecUnscaled; difVecUnscaled.reserve((size_t)3 * nF);

    for (int f = 0; f < nF; ++f) {
        int v0 = mesh.F(f, 0);
        int v1 = mesh.F(f, 1);
        int v2 = mesh.F(f, 2);
        rowsL.push_back(v0); rowsL.push_back(v1); rowsL.push_back(v2);
        colsL.push_back(v1); colsL.push_back(v2); colsL.push_back(v0);
        Eigen::Vector3d p0 = mesh.V.row(v0);
        Eigen::Vector3d p1 = mesh.V.row(v1);
        Eigen::Vector3d p2 = mesh.V.row(v2);
        // Note the reference ordering:  e01 = p0 - p1  (points FROM p1 TO p0).
        Eigen::Vector3d e01 = p0 - p1;
        Eigen::Vector3d e12 = p1 - p2;
        Eigen::Vector3d e20 = p2 - p0;

        // Face direction v -> 3D, then w^perp = v x n, normalized.
        double vx = v[2*f], vy = v[2*f + 1];
        Eigen::Vector3d faceVec = vx * frames[f].e1 + vy * frames[f].e2;
        faceVec = faceVec.cross(frames[f].n);
        double fn = faceVec.norm();
        if (fn > 1e-30) faceVec /= fn;
        else            faceVec.setZero();

        difVecUnscaled.push_back(e01.dot(faceVec));
        difVecUnscaled.push_back(e12.dot(faceVec));
        difVecUnscaled.push_back(e20.dot(faceVec));
    }

    // Build the (vertex-vertex) TP adjacency with 1 per directed edge.
    // TP is used once to compute per-vertex degrees; symmetrized.
    {
        std::vector<Trip> tri;
        tri.reserve(rowsL.size());
        for (size_t i = 0; i < rowsL.size(); ++i) {
            tri.emplace_back(rowsL[i], colsL[i], 1.0);
        }
        SpMat TP(nV, nV);
        TP.setFromTriplets(tri.begin(), tri.end());
        SpMat TPt = SpMat(TP.transpose());
        TP += TPt;

        // degree[v] = row-sum of TP = total count of incident directed edges
        // (each undirected edge contributes 2 across both adjacency passes).
        Vec degree = Vec::Zero(nV);
        for (int k = 0; k < TP.outerSize(); ++k) {
            for (SpMat::InnerIterator it(TP, k); it; ++it) {
                degree[(int)it.row()] += it.value();
            }
        }

        if (scales.size() != nF) scales = Vec::Ones(nF);

        // Outer Gauss-Newton alternation
        for (int iter = 0; iter < outerIters; ++iter) {
            if (progressStage >= 0) {
                reportProgress(progressStage, iter, outerIters, progressLabel);
            }

            // Scaled edge differences
            std::vector<double> difVec(rowsL.size());
            for (size_t i = 0; i < rowsL.size(); ++i) {
                difVec[i] = difVecUnscaled[i] * scales[(int)(i / 3)];
            }

            // Assemble the real 2n x 2n connection Laplacian:
            //   Amat[2r..2r+1, 2c..2c+1] += R(dif)      // 2x2 rotation block
            //   symmetrize
            //   Lmat = diag(degree, broadcast to 2 channels) - Amat
            std::vector<Trip> Atri;
            Atri.reserve(4 * rowsL.size());
            for (size_t i = 0; i < rowsL.size(); ++i) {
                double c = std::cos(difVec[i]);
                double s = std::sin(difVec[i]);
                int r = rowsL[i];
                int cc = colsL[i];
                Atri.emplace_back(2*r,     2*cc,      c);
                Atri.emplace_back(2*r,     2*cc + 1, -s);
                Atri.emplace_back(2*r + 1, 2*cc,      s);
                Atri.emplace_back(2*r + 1, 2*cc + 1,  c);
            }
            SpMat Amat(2 * nV, 2 * nV);
            Amat.setFromTriplets(Atri.begin(), Atri.end());
            SpMat AmatT = SpMat(Amat.transpose());
            Amat += AmatT;

            std::vector<Trip> Ltri;
            Ltri.reserve(2 * nV);
            for (int i = 0; i < 2 * nV; ++i) {
                Ltri.emplace_back(i, i, degree[i / 2]);
            }
            SpMat Lmat(2 * nV, 2 * nV);
            Lmat.setFromTriplets(Ltri.begin(), Ltri.end());
            Lmat -= Amat;
            Lmat.makeCompressed();

            // Regularize lightly to avoid singular solves on totally
            // disconnected components.
            {
                std::vector<Trip> rtri;
                rtri.reserve(2 * nV);
                for (int i = 0; i < 2 * nV; ++i) rtri.emplace_back(i, i, 1e-8);
                SpMat R(2 * nV, 2 * nV);
                R.setFromTriplets(rtri.begin(), rtri.end());
                Lmat += R;
            }

            // Inverse power iteration on Lmat (smallest eigenvector).
            Eigen::SimplicialLDLT<SpMat> solverL;
            solverL.compute(Lmat);
            if (solverL.info() != Eigen::Success) {
                std::fprintf(stderr, "[gn] Lmat LDLT failed at outer iter %d\n", iter);
                return;
            }
            Vec eigenVec(2 * nV);
            std::srand(0);
            eigenVec.setRandom();
            eigenVec /= std::max(eigenVec.norm(), 1e-30);
            for (int p = 0; p < powerIters; ++p) {
                eigenVec = solverL.solve(eigenVec);
                eigenVec /= std::max(eigenVec.norm(), 1e-30);
            }

            // Extract theta per vertex from the (cos, sin) pair.
            for (int i = 0; i < nV; ++i) {
                double c = eigenVec[2*i];
                double sn = eigenVec[2*i + 1];
                double nrm = std::sqrt(c*c + sn*sn);
                if (nrm < 1e-30) { theta[i] = 0; continue; }
                double a = std::acos(c / nrm);
                if (sn < 0) a = -a;
                theta[i] = a;
            }

            // Per-face scale update (1x1 LS solve per face).
            for (int f = 0; f < nF; ++f) {
                double num = 0, den = 0;
                for (int e = 0; e < 3; ++e) {
                    int r = rowsL[3*f + e];
                    int cc = colsL[3*f + e];
                    double pred = theta[r] - theta[cc];
                    if (pred >  M_PI) pred -= 2 * M_PI;
                    if (pred < -M_PI) pred += 2 * M_PI;
                    double u = difVecUnscaled[3*f + e];
                    num += pred * u;
                    den += u * u;
                }
                if (den > 1e-30) scales[f] = num / den;
            }
        }
    }
}

} // namespace wgf
