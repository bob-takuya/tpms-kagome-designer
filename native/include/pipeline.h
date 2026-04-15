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
#include "curl_local_integration.h"
#include "gn_global_integration.h"
#include "paired_rounding.h"
#include "progress.h"

namespace wgf {

struct PipelineOptions {
    // Browser-friendly defaults. The paper recommends lambdaInit=1000,
    // lambdaMin→0, alg1MaxIter~50 and jointIters=10, which produces
    // excellent results but takes minutes of CG solves in WASM. These
    // smaller numbers converge fast enough for interactive use while
    // still running the same algorithms end-to-end.
    double lambdaInit     = 10.0;
    double lambdaMin      = 1e-2;
    int    alg1MaxIter    = 20;
    double alg1Tol        = 1e-5;
    double mu             = 1e-4;
    int    jointIters     = 4;
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

    reportProgress(STAGE_INIT, 0, 1, "Building face frames");
    auto frames = buildFaceFrames(base);

    // 1. 6-RoSy (Knoeppel 2013). We spend 150 inverse-power-iteration
    // steps here because the cover build (§3) is extremely sensitive
    // to the smoothness of the resulting phi: σ rounding across
    // interior edges introduces spurious holonomy whenever (φ_i − φ_j + β)
    // is far from an integer multiple of π/3, which in turn flags
    // almost every vertex as a branch point and produces a useless
    // fragmented cover. A well-converged eigenvector keeps the
    // residual small enough for rounding to give a clean σ field.
    reportProgress(STAGE_ROSY, 0, 1, "Knoeppel 2013 6-RoSy eigensolve");
    SpMat L6 = build6RoSyReal(base, frames);
    Vec   M6 = face6RoSyMass(frames);
    Vec   z  = solve6RoSy(L6, M6, 150, 1e-4);
    Vec   phi = faceBaseAngleFromZ(z);
    Vec   w0base = initFaceFieldFromRosy(phi);

    // 2. Algorithm 1 on the base (for diagnostics + to warm up the cover init)
    reportProgress(STAGE_ALG1_BASE, 0, 1, "Algorithm 1 (base)");
    Alg1Result R1 = runAlg1(base, frames, w0base, {},
                            opt.lambdaInit, opt.lambdaMin,
                            opt.alg1MaxIter, opt.alg1Tol,
                            STAGE_ALG1_BASE, "Algorithm 1 (base)");
    R.alg1BaseInitCurl  = R1.initialCurl;
    R.alg1BaseFinalCurl = R1.finalCurl;
    R.alg1BaseIters     = R1.iterations;

    if (!opt.useCover) {
        // Single-family pipeline (debug path).
        reportProgress(STAGE_COMPONENT, 0, 1, "Algorithm 2 + Eq. 8 (base mesh)");
        JointResult J = solveJointScalar(base, frames, R1.w, opt.mu, opt.jointIters);
        reportProgress(STAGE_ASSEMBLE, 0, 1, "Extracting isolines");
        auto segs = extractIsolines(base, J.theta);
        for (auto& s : segs) {
            ProjectedSegment p;
            p.a = s.a; p.b = s.b;
            p.baseFaceIdx = s.faceIdx;
            p.family = 0;
            R.segments.push_back(p);
            R.numSegmentsFam[0]++;
        }
        reportProgress(STAGE_DONE, 1, 1, "Done");
        return R;
    }

    // 3. 6-fold branched cover construction (Vekhter §5.1).
    //
    // The cover is built from the ALGORITHM-1-OPTIMIZED direction field
    // (rather than the raw 6-RoSy eigenvector) so that the per-edge σ
    // rounding is stable: on a well-converged Alg1 output, the angular
    // residual (phi_i − phi_j + β) is far from the π/6 decision boundary
    // almost everywhere, and σ rounds to 0 on all non-singular edges.
    //
    // Holonomy is invariant under per-face representative choice
    // (telescoping cancellation around any closed cycle), so atan2 of
    // the Alg1 vector field can be used directly without wrapping into
    // (−π/6, π/6].
    reportProgress(STAGE_COVER_BUILD, 0, 1, "Building 6-fold branched cover");
    Vec phiRefined(base.nF());
    for (int f = 0; f < base.nF(); ++f) {
        phiRefined[f] = std::atan2(R1.w[2*f + 1], R1.w[2*f]);
    }
    auto cov = buildBranchedCover(base, frames, phiRefined);
    R.coverV = cov.mesh.nV();
    R.coverF = cov.mesh.nF();
    R.numSingular = cov.numSingular;

    // 4. Connected components of the cover.
    reportProgress(STAGE_COVER_SPLIT, 0, 1, "Labeling cover components");
    std::vector<int> comp;
    int nComp = labelComponents(cov.mesh, comp);
    R.numComponents = nComp;

    // 5. Per-component scalar field integration (Vekhter §4).
    //
    // For each connected component of the cover:
    //   (a) Initialize the per-face direction field w from the base-face
    //       layer angle (atan2(w_base) + layer * π/3, parallel-transported
    //       into the sub-mesh's frame).
    //   (b) Run per-component Algorithm 1 (curl-free projection with a
    //       single λ level) to refine w on this component.
    //   (c) Call CurlLocalIntegration to obtain the initial scale s per
    //       face (generalized eigenproblem with face Laplacian regulariser).
    //   (d) Rescale s anti-aliasing: s *= π / avgEdgeLen / maxS * userScale
    //       (matches the reference's `s_scale` step in integrateField).
    //   (e) Call GNGlobalIntegration to obtain theta per cover vertex
    //       (alternating Gauss-Newton inverse power iteration + 1x1 scale
    //       update, matching the reference's `globallyIntegrateOneComponent`
    //       verbatim).
    //   (f) Write theta and s back into global cover arrays.
    Vec thetaCover = Vec::Zero(cov.mesh.nV());
    Vec scalesCover = Vec::Zero(cov.mesh.nF());

    // Average base-mesh edge length, used by the anti-aliasing rescale.
    double avgEdgeLen = 0;
    {
        int cnt = 0;
        for (const auto& e : base.eInt) {
            int h = e.first;
            int va = base.heStart(h);
            int vb = base.he[h].vertex;
            avgEdgeLen += (base.V.row(vb) - base.V.row(va)).norm();
            cnt++;
        }
        if (cnt > 0) avgEdgeLen /= cnt;
        if (avgEdgeLen < 1e-30) avgEdgeLen = 1.0;
    }

    for (int c = 0; c < nComp; ++c) {
        reportProgress(STAGE_COMPONENT, c, nComp, "Cover: Curl-local + GN-global");
        SubMesh S = extractComponent(cov.mesh, comp, c);
        if (S.mesh.nV() < 3 || S.mesh.nF() < 1) continue;

        auto Sframes = buildFaceFrames(S.mesh);

        // (a) Initial direction field on this component, inherited from
        //     the base-face layer angle.
        Vec wS(2 * S.mesh.nF());
        for (int f = 0; f < S.mesh.nF(); ++f) {
            int covFace = S.oldFaceIdx[f];
            int bf  = cov.coverFaceBaseF[covFace];
            int lay = cov.coverFaceLayer[covFace];
            double ang = phiRefined[bf] + lay * M_PI / 3.0;
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

        // (b) Per-component Alg 1 to sharpen the direction field.
        Alg1Result RS = runAlg1(S.mesh, Sframes, wS, {},
                                /*lambdaInit=*/1.0,
                                /*lambdaMin =*/1e-2,
                                /*maxIter   =*/8,
                                opt.alg1Tol);
        R.alg1CoverFinalCurl += RS.finalCurl;

        // (c) Curl local integration → initial s.
        Vec sCompRaw;
        curlLocalIntegration(S.mesh, Sframes, RS.w,
                             /*sreg=*/1e-4, sCompRaw);

        // (d) Anti-aliasing rescale (matches CoverMesh::integrateField).
        double maxS = 0;
        for (int i = 0; i < sCompRaw.size(); ++i) {
            double a = std::fabs(sCompRaw[i]);
            if (a > maxS) maxS = a;
        }
        if (maxS < 1e-30) maxS = 1.0;
        double sScale = M_PI / avgEdgeLen / maxS;
        sCompRaw *= opt.userScale * sScale;

        // (e) GN global integration → theta.
        Vec thetaComp;
        gnGlobalIntegration(S.mesh, Sframes, RS.w, sCompRaw, thetaComp,
                            /*outerIters=*/6,
                            /*powerIters=*/20,
                            STAGE_COMPONENT,
                            "Cover: GN global integration");

        // (f) Scatter back into global cover vectors.
        for (int sv = 0; sv < S.mesh.nV(); ++sv) {
            int cv = S.oldVertexIdx[sv];
            thetaCover[cv] = thetaComp[sv];
        }
        for (int f = 0; f < S.mesh.nF(); ++f) {
            int cf = S.oldFaceIdx[f];
            scalesCover[cf] = sCompRaw[f];
        }
    }

    // 6. Paired cover rounding (Vekhter §5.2 step 3).
    //
    // Enforces that antipodal layer pairs (k, k+3) have theta values
    // satisfying  theta(v1) + theta(v2) = -offset (mod phase) so that
    // projected stripes from the two halves of each kagome line
    // interleave cleanly at the half-period offset.
    {
        int numISOLines = std::max(4, (int)std::round(8.0 * opt.userScale));
        pairedRoundTheta(cov.mesh, cov, thetaCover, numISOLines);

        // 7. Extract angular isolines on the cover. The tracer produces
        // chains (groups of segments with shared chainId) that follow a
        // level set from face to face across the cover.
        reportProgress(STAGE_ASSEMBLE, 0, 1, "Extracting isolines");
        auto segs = extractAngularIsolines(cov.mesh, thetaCover, numISOLines);

        // 8. Project each cover chain back to the base as a single
        // contiguous ribbon.
        //
        // Key insight: a single cover-level-set chain can traverse
        // multiple cover layers (because ~80% of cover edges have
        // sigma mod 3 != 0). If we split it per-segment by
        // (layer mod 3) it fragments into unconnected chunks in
        // different family buckets, which is the "grass" effect.
        //
        // Instead, we keep each chain in ONE family bucket — the
        // dominant layer mod 3 in that chain (majority vote over its
        // segments). Within that bucket the segments share endpoints
        // pairwise by construction, so kagome.ts's endpoint-matching
        // stitcher builds the chain back into a continuous polyline.
        // The "3 families at 60°" interpretation is lost (a family-0
        // ribbon can locally be at any angle as it crosses layers),
        // but the topological correctness of the cover pipeline is
        // preserved.
        //
        // Group segments by chainId first.
        std::vector<std::vector<int>> chainSegs;
        {
            int maxCid = -1;
            for (const auto& s : segs) if (s.chainId > maxCid) maxCid = s.chainId;
            chainSegs.resize(maxCid + 1);
            for (int i = 0; i < (int)segs.size(); ++i)
                chainSegs[segs[i].chainId].push_back(i);
        }
        for (const auto& chainIdxs : chainSegs) {
            if (chainIdxs.empty()) continue;
            // Majority vote for the chain's family.
            int famCount[3] = {0, 0, 0};
            for (int si : chainIdxs) {
                int cf = segs[si].faceIdx;
                int lay = cov.coverFaceLayer[cf];
                famCount[lay % 3]++;
            }
            int fam = 0;
            if (famCount[1] > famCount[fam]) fam = 1;
            if (famCount[2] > famCount[fam]) fam = 2;

            // Emit every segment in this chain under the chosen family,
            // preserving the tracer's face-to-face order.
            for (int si : chainIdxs) {
                const auto& s = segs[si];
                int cf = s.faceIdx;
                int bf = cov.coverFaceBaseF[cf];
                ProjectedSegment p;
                p.a = s.a;
                p.b = s.b;
                p.baseFaceIdx = bf;
                p.family = fam;
                R.segments.push_back(p);
                R.numSegmentsFam[fam]++;
            }
        }
    }

    reportProgress(STAGE_DONE, 1, 1, "Done");
    return R;
}

} // namespace wgf
