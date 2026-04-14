// pipeline.h — end-to-end Vekhter 2019 pipeline composition.
//
// Input:  base triangle mesh (V, F)
// Output: list of isoline segments on the base mesh, each carrying a
//         kagome family label (0, 1, 2).
//
// Stages (paper §5.2):
//   1. Build base half-edge mesh + frames
//   2. 6-RoSy eigensolve (Knöppel 2013) → per-face canonical angle φ
//   3. Initial face field from φ
//   4. Algorithm 1 on the base mesh (λ sharpening) — this is primarily
//      to produce a cleaner ŵ for initializing the cover, and also
//      produces diagnostic curl numbers.
//   5. Build the 6-fold branched cover from φ
//   6. Split the cover into connected components
//   7. For each component: lift φ, run Algorithm 1, run Algorithm 2 +
//      Eq. 8 joint (s, θ) optimization
//   8. Extract θ isolines per component, accumulate in a single list
//   9. Project each cover segment to its base face / family

#pragma once

#include "mesh.h"
#include "rosy.h"
#include "curl.h"
#include "alg1.h"
#include "alg2.h"
#include "cover.h"
#include "isolines.h"

namespace wgf {

struct PipelineOptions {
    double lambdaInit     = 1000.0;
    double lambdaMin      = 1e-3;
    int    alg1MaxIter    = 50;
    double alg1Tol        = 1e-5;
    double mu             = 1e-4;
    int    jointIters     = 10;
    double userScale      = 1.0;    // stripe density multiplier on top of π-rescale
    bool   useCover       = true;
};

struct PipelineResult {
    std::vector<ProjectedSegment> segments;
    // Diagnostics
    int    baseV = 0, baseF = 0, baseE = 0;
    int    coverV = 0, coverF = 0, numComponents = 0;
    int    numSingular = 0;
    double alg1BaseInitCurl  = 0;
    double alg1BaseFinalCurl = 0;
    int    alg1BaseIters     = 0;
    double alg1CoverFinalCurl = 0;  // averaged or total over components
    int    numSegmentsFam[3] = {0, 0, 0};
};

inline PipelineResult runPipeline(const Mesh& base, const PipelineOptions& opt) {
    PipelineResult R;
    R.baseV = base.nV();
    R.baseF = base.nF();
    R.baseE = base.nE();

    auto frames = buildFaceFrames(base);

    // 1. 6-RoSy
    SpMat L6 = build6RoSyReal(base, frames);
    Vec   M6 = face6RoSyMass(frames);
    Vec   z  = solve6RoSy(L6, M6, 30, 1e-4);
    Vec   phi = faceBaseAngleFromZ(z);
    Vec   w0base = initFaceFieldFromRosy(phi);

    // 2. Algorithm 1 on the base (for diagnostics + to warm up the cover init)
    Alg1Result R1 = runAlg1(base, frames, w0base, {},
                            opt.lambdaInit, opt.lambdaMin,
                            opt.alg1MaxIter, opt.alg1Tol);
    R.alg1BaseInitCurl  = R1.initialCurl;
    R.alg1BaseFinalCurl = R1.finalCurl;
    R.alg1BaseIters     = R1.iterations;

    if (!opt.useCover) {
        // Single-family pipeline: run Alg 2 + eq 8 on the base.
        JointResult J = solveJointScalar(base, frames, R1.w, opt.mu, opt.jointIters);
        auto segs = extractIsolines(base, J.theta);
        for (auto& s : segs) {
            ProjectedSegment p;
            p.a = s.a; p.b = s.b;
            p.baseFaceIdx = s.coverFaceIdx;
            p.family = 0;
            R.segments.push_back(p);
            R.numSegmentsFam[0]++;
        }
        return R;
    }

    // 3. 6-fold branched cover
    auto cov = buildBranchedCover(base, frames, phi);
    R.coverV = cov.mesh.nV();
    R.coverF = cov.mesh.nF();
    R.numSingular = cov.numSingular;

    // 4. Connected components
    std::vector<int> comp;
    int nComp = labelComponents(cov.mesh, comp);
    R.numComponents = nComp;

    // 5. Per-component pipeline
    for (int c = 0; c < nComp; ++c) {
        SubMesh S = extractComponent(cov.mesh, comp, c);
        if (S.mesh.nV() < 3 || S.mesh.nF() < 1) continue;

        auto Sframes = buildFaceFrames(S.mesh);

        // Initialize w on this component from the base φ via the layer.
        // For each face in the sub-mesh, look up the original cover face,
        // its base face, and layer; compute the 3D direction; project
        // into this sub-face's frame.
        Vec wS(2 * S.mesh.nF());
        for (int f = 0; f < S.mesh.nF(); ++f) {
            int covFace = S.oldFaceIdx[f];
            int bf  = cov.coverFaceBaseF[covFace];
            int lay = cov.coverFaceLayer[covFace];
            double ang = phi[bf] + lay * M_PI / 3.0;
            Eigen::Vector3d d3 =
                std::cos(ang) * frames[bf].e1 +
                std::sin(ang) * frames[bf].e2;
            double u = d3.dot(Sframes[f].e1);
            double v = d3.dot(Sframes[f].e2);
            double Ln = std::sqrt(u*u + v*v);
            if (Ln < 1e-12) { u = 1; v = 0; Ln = 1; }
            wS[2*f]     = u / Ln;
            wS[2*f + 1] = v / Ln;
        }

        Alg1Result RS = runAlg1(S.mesh, Sframes, wS, {},
                                opt.lambdaInit, opt.lambdaMin,
                                opt.alg1MaxIter, opt.alg1Tol);
        R.alg1CoverFinalCurl += RS.finalCurl;

        // Joint (s, θ) optimization
        JointResult J = solveJointScalar(S.mesh, Sframes, RS.w,
                                         opt.mu, opt.jointIters);

        // 6. Extract isolines on this sub-component
        auto localSegs = extractIsolines(S.mesh, J.theta);

        // Project each segment to base via the cover mapping
        for (const auto& s : localSegs) {
            int covFace = S.oldFaceIdx[s.coverFaceIdx];
            int bf  = cov.coverFaceBaseF[covFace];
            int lay = cov.coverFaceLayer[covFace];
            ProjectedSegment p;
            p.a = s.a;
            p.b = s.b;
            p.baseFaceIdx = bf;
            p.family = lay % 3;
            R.segments.push_back(p);
            R.numSegmentsFam[p.family]++;
        }
    }
    return R;
}

} // namespace wgf
