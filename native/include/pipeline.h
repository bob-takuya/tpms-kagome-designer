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
#include "paired_rounding.h"
#include "post_process.h"
#include "face_topology.h"
#include "progress.h"

#include <set>
#include <cstddef>
#include <climits>

namespace wgf {

struct PipelineOptions {
    // Defaults aligned with the implementation spec (Vekhter et al. 2019).
    double lambdaInit     = 1000.0;
    double lambdaMin      = 1e-3;
    int    alg1MaxIter    = 50;
    double alg1Tol        = 1e-5;
    double mu             = 1e-4;
    int    jointIters     = 10;
    double userScale      = 1.0;    // stripe density multiplier on top of π-rescale
    bool   useCover       = true;

    // Fast diagnostic mode. Skips paired rounding + per-component
    // Algorithm 2 / Eq.8 solve; replaces the cover theta with a synthetic
    // field that shares the same layer-wrap structure as the real output,
    // so extractAngularIsolines runs on a realistic cover.
    //   diagOnly = 0 : normal full pipeline (default)
    //   diagOnly = 1 : skip paired rounding + per-component solve
    //                  (uses base-vertex atan2 + layer * π/3 as theta)
    //   diagOnly = 2 : BROKEN — per-face phi assigned to vertices is
    //                  face-choice-dependent (see runPipeline for a full
    //                  explanation). Reserved for a future correct
    //                  implementation using vertex 6-RoSy or per-vertex
    //                  averaged phi. Use diagOnly=1 instead.
    int    diagOnly       = 0;
    // Cap on input vertex count; 0 = no cap. Only issues a warning if
    // exceeded (we don't mesh-decimate here, to keep the diagnostic run
    // comparable to the full run).
    int    maxVerts       = 0;

    // --- Post-processing (Vekhter §5.2) ------------------------------
    // Step 1 of the paper's post-processing pipeline: resample isolines
    // so that all segments are approximately the same length.
    //
    //   resample = true  : run the resample pass on R.segments before
    //                      returning (default).
    //   resample = false : emit raw traced segments (debug / compare).
    //   resampleLength   : target segment length. <= 0 means "auto"
    //                      (use the base mesh's mean edge length).
    //   resampleNoFastPath : disable the short-chain fast-path in the
    //                      resampler. For ablation / debugging only.
    bool   resample           = true;
    double resampleLength     = 0.0;
    bool   resampleNoFastPath = false;

    // Step 2 + 3: cut at high-curvature points, prune short segments.
    //
    // Both run after resample and use resampleLength as the scale for
    // the adaptive threshold + absolute caps + prune length floor.
    //
    //   curvatureCut         : enable the cut pass (default on).
    //   cutThresholdAbs      : absolute kappa threshold (1/length).
    //                          > 0 overrides the per-chain adaptive rule.
    //                          <= 0 means "adaptive" (per-chain p-th
    //                          percentile clamped by absolute caps).
    //   cutPercentile        : per-chain percentile used when the
    //                          threshold is adaptive (default 0.95).
    //   shortPrune           : enable the prune pass (default on).
    //   minChainLengthAbs    : absolute minimum chain arclength.
    //                          > 0 overrides the multiplier-based rule.
    //                          <= 0 means L_target · minChainMult.
    //   minChainMult         : multiplier on L_target for the minimum
    //                          chain length (default 3.0).
    bool   curvatureCut      = true;
    double cutThresholdAbs   = 0.0;
    double cutPercentile     = 0.95;
    bool   shortPrune        = true;
    double minChainLengthAbs = 0.0;
    double minChainMult      = 3.0;

    // Phase E-3 step 4 preparation: detect open-polyline endpoints so that
    // the distribution of candidate "extend to crossings" attachment points
    // can be observed ahead of wiring up the extension pass. The detection
    // stage never mutates R.segments; the SEG output stays bit-identical to
    // a run with detectLooseEnds=false (regression check).
    bool   detectLooseEnds   = true;

    // Phase E-3a-3: face-walk each loose end as a geodesic proxy and record
    // the extension path. Like detectLooseEnds, this stage is observation-
    // only — extensionPaths feeds diagnostics, never R.segments. Crossing
    // detection / trim is reserved for PR E-3a-4.
    //
    //   extendLooseEnds  : enable the extension pass (default on; gated by
    //                      detectLooseEnds since it consumes the loose-end
    //                      list).
    //   noExtend         : hard skip; mirrors the --no-extend regression
    //                      switch on the CLI.
    //   extendMaxMult    : default extension budget = extendMaxMult * L_target.
    //                      2.5 was picked from the PR #32 distance histogram:
    //                      <2L covers 97.5% of loose-end → nearest-different-
    //                      family-segment distances, plus a 25% safety margin
    //                      to account for face-walk detours over a 3D direct
    //                      line.
    //   extendMaxLength  : explicit absolute extension budget (overrides
    //                      extendMaxMult when > 0).
    bool   extendLooseEnds = true;
    bool   noExtend        = false;
    double extendMaxMult   = 2.5;
    double extendMaxLength = 0.0;
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

    // Resample pass (Vekhter §5.2 step 1). Populated when opt.resample
    // is true; left zero-initialised otherwise.
    bool          resampleApplied = false;
    ResampleStats resampleStats;

    // Cut + prune pass (Vekhter §5.2 steps 2 & 3). Populated when
    // opt.curvatureCut or opt.shortPrune is true; left zero-initialised
    // otherwise. The two steps share a stats struct because their
    // diagnostics are reported together.
    bool           cutPruneApplied = false;
    CutPruneStats  cutPruneStats;

    // Loose-end detection (Phase E-3 observation-only). Populated when
    // opt.detectLooseEnds is true; the list is just the start/end of every
    // open polyline in the post-cut+prune output. SEG segments are not
    // modified by this pass.
    bool                  looseEndsDetected = false;
    std::size_t           looseEndCount     = 0;
    std::vector<LooseEnd> looseEnds;

    // Loose-end extension (Phase E-3a-3 observation-only). Populated when
    // opt.extendLooseEnds is true and detectLooseEnds ran. Each entry is the
    // face-walking proxy path for the corresponding LooseEnd (entries are
    // parallel to looseEnds[]). R.segments is not modified.
    bool                       extensionApplied = false;
    double                     extensionMaxLength = 0.0;
    std::vector<ExtensionPath> extensionPaths;
};

inline PipelineResult runPipeline(const Mesh& baseIn, const PipelineOptions& opt) {
    // Marching cubes occasionally emits zero-area triangles along voxel
    // boundaries; they propagate into the Algorithm-1 KKT diagonal as exact
    // zeros and break SparseLU. Strip them once up front.
    Mesh base = baseIn;
    int dropped = removeDegenerateFaces(base);
    if (dropped > 0) {
        std::fprintf(stderr,
            "[mesh] dropped %d degenerate face(s) (area < 1e-14)\n", dropped);
        buildHalfEdges(base);
    }

    PipelineResult R;
    R.baseV = base.nV();
    R.baseF = base.nF();
    R.baseE = base.nE();

    if (opt.maxVerts > 0 && base.nV() > opt.maxVerts) {
        std::fprintf(stderr,
            "[pipeline] warning: base.nV=%d exceeds maxVerts=%d; running anyway\n",
            base.nV(), opt.maxVerts);
    }

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
        JointResult J = solveJointScalar(base, frames, R1.w, opt.mu, opt.jointIters, opt.userScale);
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

    // Diagnostic: boundary structure of base and cover meshes. Used to
    // classify whether 1-crossing faces and noExit stops inside
    // extractAngularIsolines are driven by the mesh boundary or by
    // interior topology/theta issues.
    {
        BoundaryStats bs = computeBoundaryStats(base);
        std::fprintf(stderr,
            "[mesh-diag] base: bdry_edges=%d bdry_verts=%d bdry_faces=%d bdry_loops=%d\n",
            bs.edges, bs.verts, bs.faces, bs.loops);
        BoundaryStats cs = computeBoundaryStats(cov.mesh);
        std::fprintf(stderr,
            "[mesh-diag] cover: bdry_edges=%d bdry_verts=%d bdry_faces=%d bdry_loops=%d\n",
            cs.edges, cs.verts, cs.faces, cs.loops);
    }

    // --- Fast diagnostic path -----------------------------------------
    //
    // Skip everything after the cover build; fabricate a theta that has
    // the same layer-wrap structure as a real run and call
    // extractAngularIsolines once so its diagnostic logs fire. Useful for
    // iterating on the tracer without paying for the 27k-variable
    // paired_rounding Gauss-Seidel + per-component Alg2/Eq.8 solves.
    if (opt.diagOnly >= 1) {
        int numISOLines = 1;
        Vec thetaCover(cov.mesh.nV());
        for (int v = 0; v < cov.mesh.nV(); ++v) {
            int baseV = cov.coverVertexBaseV[v];
            int layer = cov.coverVertexLayer[v];
            double thetaBase;
            if (opt.diagOnly >= 2) {
                // WARNING: diagOnly=2 is BROKEN as currently implemented.
                //
                // `phi` is per-FACE (6-RoSy canonical angle in (-π/6, π/6]);
                // neighbouring faces at the same vertex differ by ≈ k·π/3
                // (the 6-RoSy rotational ambiguity). Picking "any incident
                // face" therefore produces a vertex-valued θ that is
                // face-choice-dependent — you get a different θ for the
                // same cover vertex depending on which base face we happen
                // to visit via v2he, which breaks the per-cover-face wrap
                // structure that diagOnly=1 preserves.
                //
                // A correct implementation would need one of:
                //   (a) a vertex 6-RoSy phi (solve the Knoeppel eigenproblem
                //       with a vertex mass matrix), or
                //   (b) per-vertex averaging of phi across incident faces
                //       AFTER unwrapping each face's phi to within π/6 of
                //       some reference (e.g. the first face's phi).
                //
                // We keep the broken branch in place rather than removing it
                // so that the option name stays reserved and so that this
                // comment stays discoverable. diagOnly=1 is sufficient for
                // the current diagnostic (it still exercises the tracer on
                // a real layer-wrap structure, just skips paired_rounding
                // + per-component Alg2/Eq.8).
                int h = base.v2he[baseV];
                int fBase = (h >= 0) ? base.he[h].face : 0;
                thetaBase = phi[fBase];
            } else {
                thetaBase = std::atan2(base.V(baseV, 1), base.V(baseV, 0));
            }
            double t = thetaBase + layer * M_PI / 3.0;
            while (t >  M_PI) t -= 2.0 * M_PI;
            while (t < -M_PI) t += 2.0 * M_PI;
            thetaCover[v] = t;
        }
        std::fprintf(stderr,
            "[diag-only] mode=%d: synthetic theta on cover (nV=%d), "
            "running extractAngularIsolines with %d iso(s)\n",
            opt.diagOnly, cov.mesh.nV(), numISOLines);
        if (opt.diagOnly >= 2) {
            std::fprintf(stderr,
                "[diag-only] WARNING: diagOnly=2 is known-broken "
                "(per-face phi assigned to vertex). See pipeline.h "
                "for details. Use diagOnly=1 for reliable diagnostics.\n");
        }
        reportProgress(STAGE_ASSEMBLE, 0, 1, "diag-only: extracting isolines");
        (void)extractAngularIsolines(cov.mesh, thetaCover, numISOLines);
        R.segments.clear();
        reportProgress(STAGE_DONE, 1, 1, "Done (diag-only)");
        return R;
    }

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
    //   (c) Run Algorithm 2 (eq. 7) for initial s.
    //   (d) Apply anti-aliasing rescale (π/ρ) and userScale.
    //   (e) Run Eq. 8 alternating optimization for (θ, s).
    //   (f) Write theta and s back into global cover arrays.
    Vec thetaCover = Vec::Zero(cov.mesh.nV());
    Vec scalesCover = Vec::Zero(cov.mesh.nF());

    for (int c = 0; c < nComp; ++c) {
        reportProgress(STAGE_COMPONENT, c, nComp, "Cover: Algorithm 2 + Eq. 8");
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
        // Use the same λ schedule as the base run (Vekhter et al. 2019 §4.1.3).
        Alg1Result RS = runAlg1(S.mesh, Sframes, wS, {},
                                opt.lambdaInit,
                                opt.lambdaMin,
                                opt.alg1MaxIter,
                                opt.alg1Tol);
        R.alg1CoverFinalCurl += RS.finalCurl;

        // (c,d,e) Algorithm 2 + Eq. 8 alternating optimization.
        JointResult J = solveJointScalar(
            S.mesh, Sframes, RS.w, opt.mu, opt.jointIters, opt.userScale);
        Vec sCompRaw = J.s;
        Vec thetaComp = Vec::Zero(S.mesh.nV());
        for (int sv = 0; sv < S.mesh.nV(); ++sv) {
            thetaComp[sv] = std::atan2(J.theta[2*sv + 1], J.theta[2*sv]);
        }

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
        // --- Diagnostic: per-chain family/layer composition (log only) ---
        {
            int totalChains    = 0;
            int uniqFamChains  = 0;
            int mixedFamChains = 0;
            std::size_t minLen = SIZE_MAX;
            std::size_t maxLen = 0;
            std::size_t totalLen = 0;
            int bucket1_4    = 0;
            int bucket5_20   = 0;
            int bucket21_100 = 0;
            int bucket101p   = 0;
            for (int chainId = 0; chainId < (int)chainSegs.size(); ++chainId) {
                const auto& chainIdxs = chainSegs[chainId];
                if (chainIdxs.empty()) continue;
                int famCount[3] = {0, 0, 0};
                std::set<int> uniqueLayers;
                for (int si : chainIdxs) {
                    int lay = cov.coverFaceLayer[segs[si].faceIdx];
                    famCount[lay % 3]++;
                    uniqueLayers.insert(lay);
                }
                fprintf(stderr, "[chain-diag] chainId=%d segs=%zu fam=[%d,%d,%d] "
                    "uniqLayers=%zu\n",
                    chainId, chainIdxs.size(),
                    famCount[0], famCount[1], famCount[2],
                    uniqueLayers.size());

                ++totalChains;
                int nonZeroFam = (famCount[0] > 0) + (famCount[1] > 0) + (famCount[2] > 0);
                if (nonZeroFam <= 1) ++uniqFamChains;
                else                 ++mixedFamChains;
                std::size_t L = chainIdxs.size();
                totalLen += L;
                if (L < minLen) minLen = L;
                if (L > maxLen) maxLen = L;
                if      (L <= 4)   ++bucket1_4;
                else if (L <= 20)  ++bucket5_20;
                else if (L <= 100) ++bucket21_100;
                else               ++bucket101p;
            }
            if (totalChains == 0) minLen = 0;
            double avgLen   = totalChains > 0 ? (double)totalLen / totalChains : 0.0;
            double uniqPct  = totalChains > 0 ? 100.0 * uniqFamChains  / totalChains : 0.0;
            double mixedPct = totalChains > 0 ? 100.0 * mixedFamChains / totalChains : 0.0;
            fprintf(stderr,
                "[chain-diag] summary: totalChains=%d uniqFam=%d(%.1f%%) "
                "mixedFam=%d(%.1f%%) segLen min=%zu max=%zu avg=%.1f "
                "buckets[1-4,5-20,21-100,>100]=[%d,%d,%d,%d]\n",
                totalChains, uniqFamChains, uniqPct,
                mixedFamChains, mixedPct,
                minLen, maxLen, avgLen,
                bucket1_4, bucket5_20, bucket21_100, bucket101p);
        }

        for (int chainId = 0; chainId < (int)chainSegs.size(); ++chainId) {
            const auto& chainIdxs = chainSegs[chainId];
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
                p.chainId = chainId;
                R.segments.push_back(p);
                R.numSegmentsFam[fam]++;
            }
        }
    }

    // --- Post-processing: resample + cut + prune (Vekhter §5.2, steps 1-3).
    //     Loose-end detection runs alongside; the actual extend-to-crossings
    //     pass (step 4) will follow in a later PR.
    const bool wantPostProc =
        (opt.resample || opt.curvatureCut || opt.shortPrune || opt.detectLooseEnds)
        && !R.segments.empty();
    if (wantPostProc) {
        double meshMean = meshMeanEdgeLength(base);

        // Auto-detect target length.
        //
        // Previously we used meshMeanEdge, but on typical marching-cubes
        // meshes the traced segments are ~half the mesh edge length, so a
        // mesh-edge-sized target collapsed neighbouring samples and shrank
        // total arclength. Median of the *input* segment lengths tracks the
        // tracer's native sampling density, so the resample pass now only
        // uniformizes variance rather than downsampling.
        //
        // The same L_target is reused by the cut-threshold caps and the
        // short-prune length floor, so we compute it once up front even when
        // the resample pass itself is disabled.
        double L_target;
        const char* L_source;
        if (opt.resampleLength > 0.0) {
            L_target = opt.resampleLength;
            L_source = "explicit";
        } else {
            std::vector<double> lens;
            lens.reserve(R.segments.size());
            for (const auto& s : R.segments) {
                double d = (s.b - s.a).norm();
                if (d > 0.0) lens.push_back(d);
            }
            if (!lens.empty()) {
                std::sort(lens.begin(), lens.end());
                L_target = lens[lens.size() / 2];
                L_source = "input-median";
            } else {
                L_target = meshMean;
                L_source = "mesh-mean-edge";
            }
        }

    if (opt.resample) {
        ResampleStats rs;
        auto resampled = resampleProjectedSegments(
            R.segments, L_target, rs, opt.resampleNoFastPath);
        rs.meshMeanEdge = meshMean;

        std::fprintf(stderr,
            "[resample] targetLength=%.6f source=%s meshMeanEdge=%.6f\n",
            rs.targetLength, L_source, rs.meshMeanEdge);
        std::fprintf(stderr,
            "[resample] input-chains=%d output-polylines=%d  "
            "(open=%d closed=%d)\n",
            rs.inputChains, rs.outputPolylines,
            rs.openChains, rs.closedLoops);
        std::fprintf(stderr,
            "[resample] input-segs=%d output-segs=%d\n",
            rs.inputSegs, rs.outputSegs);
        std::fprintf(stderr,
            "[resample] input-seg-length: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
            rs.inputSegLenMean, rs.inputSegLenStd,
            rs.inputSegLenMin,  rs.inputSegLenMax);
        std::fprintf(stderr,
            "[resample] output-seg-length: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
            rs.outputSegLenMean, rs.outputSegLenStd,
            rs.outputSegLenMin,  rs.outputSegLenMax);
        double diffPct = (rs.inputTotalArclen > 0.0)
            ? 100.0 * std::fabs(rs.outputTotalArclen - rs.inputTotalArclen)
                    / rs.inputTotalArclen
            : 0.0;
        std::fprintf(stderr,
            "[resample] total-arclength: input=%.6f output=%.6f  diff=%.3f%%\n",
            rs.inputTotalArclen, rs.outputTotalArclen, diffPct);

        // --- Fast-path bookkeeping (E-1.5) ------------------------------
        //
        // How many chains were preserved verbatim by the short-chain
        // fast-path vs. actually resampled, plus the ratio within each
        // open/closed class. Disabled when --resample-no-fast-path is set
        // (both counters will be zero on the fast-path side then).
        {
            std::size_t totalFast =
                rs.openFastPath + rs.closedFastPath;
            std::size_t totalAll =
                (std::size_t)rs.outputPolylines;
            double fastPct = (totalAll > 0)
                ? 100.0 * (double)totalFast / (double)totalAll
                : 0.0;
            std::fprintf(stderr,
                "[resample-fastpath] open-fastpath=%zu closed-fastpath=%zu "
                "total=%zu/%zu (%.2f%%)\n",
                rs.openFastPath, rs.closedFastPath,
                totalFast, totalAll, fastPct);
            std::fprintf(stderr,
                "[resample-counts] open-resampled=%zu open-preserved=%zu "
                "closed-resampled=%zu closed-preserved=%zu\n",
                rs.openResampled, rs.openFastPath,
                rs.closedResampled, rs.closedFastPath);
        }

        // --- Chain-level arclength diagnostics --------------------------
        //
        // Histogram the per-polyline shrink magnitude |before-after|/before
        // for open chains and closed loops separately, then print the 20
        // worst offenders. Pinpoints which subset of chains is driving the
        // total-arclength gap without having to diff full SEG dumps.
        {
            auto pctOf = [](const ResamplePolyStat& p) -> double {
                if (p.beforeLen <= 0.0) return 0.0;
                return 100.0 * std::fabs(p.beforeLen - p.afterLen) / p.beforeLen;
            };

            int openBuckets[6]   = {0, 0, 0, 0, 0, 0};
            int closedBuckets[6] = {0, 0, 0, 0, 0, 0};
            auto bucketIndex = [](double pct) -> int {
                if (pct < 0.1)  return 0;
                if (pct < 1.0)  return 1;
                if (pct < 5.0)  return 2;
                if (pct < 10.0) return 3;
                if (pct < 20.0) return 4;
                return 5;
            };
            for (const auto& p : rs.perPoly) {
                int b = bucketIndex(pctOf(p));
                if (p.isClosed) ++closedBuckets[b];
                else            ++openBuckets[b];
            }
            std::fprintf(stderr,
                "[resample-diff-hist] open:   <0.1%%=%d <1%%=%d <5%%=%d <10%%=%d <20%%=%d >=20%%=%d\n",
                openBuckets[0], openBuckets[1], openBuckets[2],
                openBuckets[3], openBuckets[4], openBuckets[5]);
            std::fprintf(stderr,
                "[resample-diff-hist] closed: <0.1%%=%d <1%%=%d <5%%=%d <10%%=%d <20%%=%d >=20%%=%d\n",
                closedBuckets[0], closedBuckets[1], closedBuckets[2],
                closedBuckets[3], closedBuckets[4], closedBuckets[5]);

            // Fast-path chains should have before/after arclength equal to
            // numerical precision (the resampler copies points verbatim,
            // and polylineArclen computes both sums with the same formula).
            // Split the histogram into "<1e-12 abs diff" vs anything else
            // so any regression in the preservation invariant surfaces
            // immediately. Counts are absolute (not percentage) because
            // fast-path chains have, by construction, tiny diffs and the
            // % bucket thresholds would all land in the <0.1% bin.
            std::size_t openFastZero = 0, openFastOther = 0;
            std::size_t closedFastZero = 0, closedFastOther = 0;
            for (const auto& p : rs.perPoly) {
                if (!p.fastPath) continue;
                double absDiff = std::fabs(p.beforeLen - p.afterLen);
                bool zero = (absDiff < 1e-12);
                if (p.isClosed) {
                    if (zero) ++closedFastZero; else ++closedFastOther;
                } else {
                    if (zero) ++openFastZero;   else ++openFastOther;
                }
            }
            std::fprintf(stderr,
                "[resample-diff-fastpath-only] open:   <1e-12=%zu others=%zu\n",
                openFastZero, openFastOther);
            std::fprintf(stderr,
                "[resample-diff-fastpath-only] closed: <1e-12=%zu others=%zu\n",
                closedFastZero, closedFastOther);

            std::vector<int> order(rs.perPoly.size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
            std::sort(order.begin(), order.end(),
                [&](int i, int j) {
                    return pctOf(rs.perPoly[i]) > pctOf(rs.perPoly[j]);
                });
            const int K = std::min<int>(20, (int)order.size());
            std::fprintf(stderr, "[resample-diff-worst]\n");
            for (int k = 0; k < K; ++k) {
                const auto& p = rs.perPoly[order[k]];
                std::fprintf(stderr,
                    "  chainId=%d family=%d closed=%d before=%.6f after=%.6f diff=%.3f%%\n",
                    p.chainId, p.family, p.isClosed ? 1 : 0,
                    p.beforeLen, p.afterLen, pctOf(p));
            }
        }

        R.segments = std::move(resampled);
        R.numSegmentsFam[0] = R.numSegmentsFam[1] = R.numSegmentsFam[2] = 0;
        for (const auto& s : R.segments) {
            if (s.family >= 0 && s.family < 3) R.numSegmentsFam[s.family]++;
        }
        R.resampleApplied = true;
        R.resampleStats   = rs;
    } // end if (opt.resample)

    // --- Cut + prune pass (Vekhter §5.2 steps 2 & 3) ---------------------
    //
    // Runs on whatever R.segments currently holds (resampled output if the
    // previous step ran, raw tracer output otherwise). Uses the same
    // L_target as the resample pass so that the adaptive threshold caps and
    // the minimum chain length stay scale-aligned with the nominal segment
    // size of the ribbon. When BOTH passes are disabled, R.segments is
    // left untouched (regression-check path).
    if (opt.curvatureCut || opt.shortPrune) {
        CutPruneOptions cpOpt;
        cpOpt.cutThresholdAbs   = opt.cutThresholdAbs;
        cpOpt.cutPercentile     = opt.cutPercentile;
        cpOpt.disableCut        = !opt.curvatureCut;
        cpOpt.minChainLengthAbs = opt.minChainLengthAbs;
        cpOpt.minChainMult      = opt.minChainMult;
        cpOpt.disablePrune      = !opt.shortPrune;
        cpOpt.L_target          = L_target;

        CutPruneStats cps;
        auto cpOut = cutAndPruneProjectedSegments(R.segments, cpOpt, cps);

        std::fprintf(stderr,
            "[postproc-cut] method=%s p=%.3f L_target=%.6f disableCut=%d disablePrune=%d\n",
            cps.cutWasAdaptive ? "adaptive-percentile" : "absolute",
            opt.cutPercentile, L_target,
            cpOpt.disableCut ? 1 : 0, cpOpt.disablePrune ? 1 : 0);
        if (cps.cutWasAdaptive && cps.thresholdSamples > 0) {
            std::fprintf(stderr,
                "[postproc-cut] threshold-stats: min=%.6f max=%.6f mean=%.6f "
                "(over %d chains)\n",
                cps.thrMin, cps.thrMax, cps.thrMean, cps.thresholdSamples);
        } else if (!cps.cutWasAdaptive) {
            std::fprintf(stderr,
                "[postproc-cut] threshold=%.6f (absolute override)\n",
                opt.cutThresholdAbs);
        }
        std::fprintf(stderr,
            "[postproc-cut] kappa-dist-input: samples=%d p50=%.6f p75=%.6f "
            "p90=%.6f p95=%.6f p99=%.6f max=%.6f\n",
            cps.totalKappaSamples,
            cps.kappaP50, cps.kappaP75, cps.kappaP90,
            cps.kappaP95, cps.kappaP99, cps.kappaMax);
        std::fprintf(stderr,
            "[postproc-cut] input-polylines=%d cuts-applied=%d "
            "output-polylines=%d\n",
            cps.inputPolylines, cps.cutsApplied, cps.afterCutPolylines);
        std::fprintf(stderr,
            "[postproc-cut] kappa-dist-output: p95=%.6f p99=%.6f max=%.6f "
            "(should sit below the input threshold band)\n",
            cps.outKappaP95, cps.outKappaP99, cps.outKappaMax);

        std::fprintf(stderr,
            "[postproc-prune] min-length=%.6f method=%s\n",
            cps.effectiveMinChainLength,
            (opt.minChainLengthAbs > 0.0) ? "absolute" : "mult-of-L_target");
        std::fprintf(stderr,
            "[postproc-prune] length-hist before (<1L,<2L,<5L,<10L,>=10L): "
            "[%d, %d, %d, %d, %d]\n",
            cps.lenHistBefore[0], cps.lenHistBefore[1], cps.lenHistBefore[2],
            cps.lenHistBefore[3], cps.lenHistBefore[4]);
        std::fprintf(stderr,
            "[postproc-prune] pruned=%d kept=%d\n",
            cps.prunedPolylines, cps.afterPrunePolylines);
        std::fprintf(stderr,
            "[postproc-prune] length-hist after  (<1L,<2L,<5L,<10L,>=10L): "
            "[%d, %d, %d, %d, %d]\n",
            cps.lenHistAfter[0], cps.lenHistAfter[1], cps.lenHistAfter[2],
            cps.lenHistAfter[3], cps.lenHistAfter[4]);

        double inArc    = std::max(cps.inputArclen,  1e-30);
        double arcDelta = 100.0 * std::fabs(cps.outputArclen - cps.inputArclen) / inArc;
        std::fprintf(stderr,
            "[postproc-summary] input=%d after-cut=%d after-prune=%d  "
            "arclen input=%.6f cut=%.6f output=%.6f diff=%.3f%%\n",
            cps.inputPolylines, cps.afterCutPolylines, cps.afterPrunePolylines,
            cps.inputArclen, cps.cutArclen, cps.outputArclen, arcDelta);
        std::fprintf(stderr,
            "[postproc-summary] family counts after: fam0=%d fam1=%d fam2=%d\n",
            cps.familyCountsAfter[0], cps.familyCountsAfter[1],
            cps.familyCountsAfter[2]);

        R.segments = std::move(cpOut);
        R.numSegmentsFam[0] = R.numSegmentsFam[1] = R.numSegmentsFam[2] = 0;
        for (const auto& s : R.segments) {
            if (s.family >= 0 && s.family < 3) R.numSegmentsFam[s.family]++;
        }
        R.cutPruneApplied = true;
        R.cutPruneStats   = cps;
    } // end if (curvatureCut || shortPrune)

    // --- Loose-end detection (Phase E-3, observation-only) -------------
    //
    // Rebuilds the post-cut+prune polyline list from R.segments and flags
    // the start/end of every OPEN polyline. detectLooseEnds does not
    // modify R.segments, so the SEG output on stdout stays bit-identical
    // to a run with --no-detect-loose-ends. Diagnostics below characterise
    // where loose ends concentrate so that the upcoming extend-to-crossings
    // pass (§5.2 step 4) can be sized / validated against the observed
    // distribution.
    if (opt.detectLooseEnds && !R.segments.empty()) {
        auto lePolys = buildOrderedPolylines(R.segments);
        auto les     = detectLooseEnds(lePolys);

        int nOpen = 0, nClosed = 0;
        for (const auto& pl : lePolys) {
            if (pl.isClosed) ++nClosed; else ++nOpen;
        }

        // Per-family and per-source (start vs end) counts.
        int perFamily[3] = {0, 0, 0};
        int fromStart = 0, fromEnd = 0;
        for (const auto& e : les) {
            if (e.family >= 0 && e.family < 3) ++perFamily[e.family];
            if (e.isStart) ++fromStart; else ++fromEnd;
        }

        // Tangent-angle histogram. Take |tangent · n| so the sign of the
        // face normal and the outward direction of the tangent drop out;
        // the angle lands in [0°, 90°], where 90° means the tangent lies
        // in the face plane (expected for on-surface loose ends).
        int angBuckets[4] = {0, 0, 0, 0};  // [<10°, <30°, <60°, <90°]
        for (const auto& e : les) {
            if (e.baseFaceIdx < 0 || e.baseFaceIdx >= base.nF()) continue;
            Eigen::Vector3d va = base.V.row(base.F(e.baseFaceIdx, 0));
            Eigen::Vector3d vb = base.V.row(base.F(e.baseFaceIdx, 1));
            Eigen::Vector3d vc = base.V.row(base.F(e.baseFaceIdx, 2));
            Eigen::Vector3d nrm = (vb - va).cross(vc - va);
            double nn = nrm.norm();
            if (nn < 1e-30) continue;
            nrm /= nn;
            double tn = std::abs(e.tangent.dot(nrm));
            if (tn > 1.0) tn = 1.0;
            double angleDeg = std::acos(tn) * 180.0 / M_PI;
            int idx = (angleDeg < 10.0) ? 0
                    : (angleDeg < 30.0) ? 1
                    : (angleDeg < 60.0) ? 2
                    :                     3;
            ++angBuckets[idx];
        }

        // Nearest different-family-segment distance. Bucket R.segments by
        // baseFaceIdx, then probe the loose end's face plus its (up to
        // three) 1-hop-adjacent faces. Limits the comparison to
        // O(N_endpoints · k) where k is the local per-face segment
        // density, vs a full O(N_endpoints · N_segments) sweep.
        std::unordered_map<int, std::vector<int>> faceToSegs;
        faceToSegs.reserve(R.segments.size() / 4 + 1);
        for (int i = 0; i < (int)R.segments.size(); ++i) {
            faceToSegs[R.segments[i].baseFaceIdx].push_back(i);
        }
        auto pointSegDist = [](const Eigen::Vector3d& p,
                               const Eigen::Vector3d& a,
                               const Eigen::Vector3d& b) {
            Eigen::Vector3d ab = b - a;
            double abab = ab.dot(ab);
            if (abab < 1e-30) return (p - a).norm();
            double t = (p - a).dot(ab) / abab;
            if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
            return (a + t * ab - p).norm();
        };
        int distBuckets[5] = {0, 0, 0, 0, 0};
        for (const auto& e : les) {
            if (e.baseFaceIdx < 0) { ++distBuckets[4]; continue; }
            double best = std::numeric_limits<double>::infinity();
            auto probeFace = [&](int f) {
                auto it = faceToSegs.find(f);
                if (it == faceToSegs.end()) return;
                for (int si : it->second) {
                    const auto& s = R.segments[si];
                    if (s.family == e.family) continue;
                    double d = pointSegDist(e.pos, s.a, s.b);
                    if (d < best) best = d;
                }
            };
            probeFace(e.baseFaceIdx);
            if (e.baseFaceIdx < base.nF()) {
                int h0 = base.f2he[e.baseFaceIdx];
                if (h0 >= 0) {
                    int h1 = base.he[h0].next;
                    int h2 = base.he[h1].next;
                    int hs[3] = { h0, h1, h2 };
                    for (int k = 0; k < 3; ++k) {
                        int nf = oppositeFaceAcross(base, hs[k], e.baseFaceIdx);
                        if (nf >= 0) probeFace(nf);
                    }
                }
            }
            if (!std::isfinite(best))                 ++distBuckets[4];
            else if (best < 0.25 * L_target)          ++distBuckets[0];
            else if (best < 0.5  * L_target)          ++distBuckets[1];
            else if (best <        L_target)          ++distBuckets[2];
            else if (best < 2.0  * L_target)          ++distBuckets[3];
            else                                      ++distBuckets[4];
        }

        std::fprintf(stderr,
            "[postproc-looseends] detected=%zu\n", les.size());
        std::fprintf(stderr,
            "[postproc-looseends] per-family: fam0=%d fam1=%d fam2=%d\n",
            perFamily[0], perFamily[1], perFamily[2]);
        std::fprintf(stderr,
            "[postproc-looseends] per-chain-source: from-start=%d from-end=%d "
            "(both should equal N_open=%d)\n",
            fromStart, fromEnd, nOpen);
        std::fprintf(stderr,
            "[postproc-looseends] tangent-angle-hist (deg from face normal, "
            "expect mass near 90deg): <10=%d <30=%d <60=%d <90=%d\n",
            angBuckets[0], angBuckets[1], angBuckets[2], angBuckets[3]);
        std::fprintf(stderr,
            "[postproc-looseends] nearest-different-family-segment-dist "
            "(L_target=%.6f): <0.25L=%d <0.5L=%d <1L=%d <2L=%d >=2L=%d\n",
            L_target,
            distBuckets[0], distBuckets[1], distBuckets[2],
            distBuckets[3], distBuckets[4]);
        std::fprintf(stderr,
            "[postproc-looseends-summary] open-polylines=%d "
            "closed-polylines=%d detected-loose-ends=%zu (expected 2*N_open=%d)\n",
            nOpen, nClosed, les.size(), 2 * nOpen);

        R.looseEndsDetected = true;
        R.looseEndCount     = les.size();
        R.looseEnds         = std::move(les);
    } // end if (detectLooseEnds)

    // --- Loose-end extension (Phase E-3a-3, observation-only) ----------
    //
    // Face-walks each detected loose end as a geodesic proxy and records
    // the extension path for diagnostics. R.segments is unchanged here:
    // crossing detection / trim and the actual chain edits happen in
    // PR E-3a-4. EXTEND_MAX_MULT defaults to 2.5 (PR #32 distance
    // histogram: <2L covers 97.5% of loose-end → nearest-different-family
    // distances; +25% margin for face-walk detours).
    if (opt.extendLooseEnds && !opt.noExtend && R.looseEndsDetected) {
        const double maxExt = (opt.extendMaxLength > 0.0)
            ? opt.extendMaxLength
            : opt.extendMaxMult * L_target;

        std::vector<ExtensionPath> paths;
        paths.reserve(R.looseEnds.size());

        // Outcome counters.
        int outMax = 0, outBoundary = 0, outRevisited = 0,
            outFailed = 0, outDegenerate = 0;
        // Per-phase DEGENERATE counters, parallel to the
        // ExtensionPath::DegenerateDiag::Phase enum (5 entries):
        //   [0] INITIAL_POS_ON_EDGE      — first computeExitPoint failed
        //                                  even after the initial nudge.
        //   [1] INITIAL_TANGENT_PARALLEL — initial tangent ∥ face normal.
        //   [2] TRANSIT_POS_ON_EDGE      — computeExitPoint failed after a
        //                                  successful face transit.
        //   [3] TRANSIT_TANGENT_PARALLEL — transported tangent collapsed to
        //                                  zero in the new face's plane.
        //   [4] OTHER                    — degenerate without a diag
        //                                  payload (e.g. invalid face id).
        int degenPhaseCounts[5] = {0, 0, 0, 0, 0};
        // Distance histogram (relative to maxExt): <0.25, <0.5, <0.75, =max.
        int distBuckets[4] = {0, 0, 0, 0};
        // Faces-visited histogram: 1-face, 2, 3, 4, 5+.
        int faceBuckets[5] = {0, 0, 0, 0, 0};
        // Per-family success counts (success == MAX_LENGTH outcome).
        int famSuccess[3] = {0, 0, 0};
        // Rotation-angle histogram across all face transits (radians);
        // buckets: <1°, <10°, <30°, ≥30°.
        int rotBuckets[4] = {0, 0, 0, 0};
        // Post-transport drift histogram (|t·n_to|) across all transits;
        // buckets: <0.01, <0.1, <0.5, ≥0.5.
        int driftBuckets[4] = {0, 0, 0, 0};
        // First N DEGENERATE cases, dumped verbatim to stderr for root-cause
        // analysis of the remaining collapses.
        constexpr int kDegenSampleLimit = 20;
        int degenSampleCount = 0;

        for (std::size_t i = 0; i < R.looseEnds.size(); ++i) {
            ExtensionPath p = extendLooseEnd(R.looseEnds[i], base, maxExt);
            p.looseEndIdx = (int)i;

            switch (p.outcome) {
                case ExtensionPath::MAX_LENGTH:     ++outMax;        break;
                case ExtensionPath::BOUNDARY:       ++outBoundary;   break;
                case ExtensionPath::REVISITED_FACE: ++outRevisited;  break;
                case ExtensionPath::FAILED:         ++outFailed;     break;
                case ExtensionPath::DEGENERATE: {
                    ++outDegenerate;
                    int phaseIdx = p.degenerateDiag.haveInfo
                        ? (int)p.degenerateDiag.phase
                        : (int)ExtensionPath::DegenerateDiag::OTHER;
                    if (phaseIdx < 0 || phaseIdx > 4) phaseIdx = 4;
                    ++degenPhaseCounts[phaseIdx];
                    break;
                }
            }

            const double r = (maxExt > 0.0) ? (p.totalDistance / maxExt) : 0.0;
            int db;
            if      (r < 0.25)  db = 0;
            else if (r < 0.5)   db = 1;
            else if (r < 0.75)  db = 2;
            else                db = 3;
            ++distBuckets[db];

            const int nFaces = (int)p.facePath.size();
            int fb;
            if      (nFaces <= 1) fb = 0;
            else if (nFaces == 2) fb = 1;
            else if (nFaces == 3) fb = 2;
            else if (nFaces == 4) fb = 3;
            else                  fb = 4;
            ++faceBuckets[fb];

            if (p.outcome == ExtensionPath::MAX_LENGTH) {
                int fam = R.looseEnds[i].family;
                if (fam >= 0 && fam < 3) ++famSuccess[fam];
            }

            // Parallel-transport diagnostics.
            for (double a : p.rotationAngles) {
                const double deg = a * (180.0 / M_PI);
                if      (deg < 1.0)  ++rotBuckets[0];
                else if (deg < 10.0) ++rotBuckets[1];
                else if (deg < 30.0) ++rotBuckets[2];
                else                 ++rotBuckets[3];
            }
            for (double d : p.tangentDrifts) {
                const double ad = std::abs(d);
                if      (ad < 0.01) ++driftBuckets[0];
                else if (ad < 0.1)  ++driftBuckets[1];
                else if (ad < 0.5)  ++driftBuckets[2];
                else                ++driftBuckets[3];
            }

            // First kDegenSampleLimit DEGENERATE cases: dump context so we
            // can tell whether the collapse is happening on the first call
            // (bad loose-end tangent) or after a transit (transport bug).
            if (p.outcome == ExtensionPath::DEGENERATE &&
                p.degenerateDiag.haveInfo &&
                degenSampleCount < kDegenSampleLimit) {
                const auto& d = p.degenerateDiag;
                const char* phaseStr;
                switch (d.phase) {
                    case ExtensionPath::DegenerateDiag::INITIAL_POS_ON_EDGE:
                        phaseStr = "initial-pos-on-edge"; break;
                    case ExtensionPath::DegenerateDiag::INITIAL_TANGENT_PARALLEL:
                        phaseStr = "initial-tangent-parallel"; break;
                    case ExtensionPath::DegenerateDiag::TRANSIT_POS_ON_EDGE:
                        phaseStr = "transit-pos-on-edge"; break;
                    case ExtensionPath::DegenerateDiag::TRANSIT_TANGENT_PARALLEL:
                        phaseStr = "transit-tangent-parallel"; break;
                    default:
                        phaseStr = "other"; break;
                }
                std::fprintf(stderr,
                    "[postproc-extend-dry-degen-sample] #%d\n"
                    "  looseEndIdx=%d chainId=%d family=%d phase=%s\n"
                    "  current_face=%d num_faces_traversed=%d\n"
                    "  current_pos=(%.9g, %.9g, %.9g)\n"
                    "  current_tangent=(%.9g, %.9g, %.9g)\n"
                    "  face_normal=(%.9g, %.9g, %.9g)\n"
                    "  tangent_dot_normal=%.9g\n"
                    "  projected_tangent=(%.9g, %.9g, %.9g) mag=%.9g\n"
                    "  point_to_vertex_dists=(%.9g, %.9g, %.9g)\n"
                    "  point_to_edge_dists=(%.9g, %.9g, %.9g)\n",
                    degenSampleCount,
                    (int)i,
                    R.looseEnds[i].chainId, R.looseEnds[i].family,
                    phaseStr,
                    d.currentFace, d.numFacesTraversed,
                    d.currentPos.x(), d.currentPos.y(), d.currentPos.z(),
                    d.currentTangent.x(), d.currentTangent.y(), d.currentTangent.z(),
                    d.faceNormal.x(), d.faceNormal.y(), d.faceNormal.z(),
                    d.tangentDotNormal,
                    d.projectedTangent.x(), d.projectedTangent.y(), d.projectedTangent.z(),
                    d.projectedMag,
                    d.pointToVertexDists[0], d.pointToVertexDists[1], d.pointToVertexDists[2],
                    d.pointToEdgeDists[0],   d.pointToEdgeDists[1],   d.pointToEdgeDists[2]);
                ++degenSampleCount;
            }

            paths.push_back(std::move(p));
        }

        std::fprintf(stderr,
            "[postproc-extend-dry] max-mult=%.3f max-length=%.6f "
            "(= %.3f x L_target)\n",
            opt.extendMaxMult, maxExt,
            (L_target > 0.0) ? (maxExt / L_target) : 0.0);
        std::fprintf(stderr,
            "[postproc-extend-dry] attempts=%zu\n", paths.size());
        std::fprintf(stderr,
            "[postproc-extend-dry] outcomes: max-length=%d boundary=%d "
            "revisited-face=%d failed=%d degenerate=%d\n",
            outMax, outBoundary, outRevisited, outFailed, outDegenerate);
        std::fprintf(stderr,
            "[postproc-extend-dry] degenerate-phase-breakdown: "
            "initial-pos-on-edge=%d initial-tangent-parallel=%d "
            "transit-pos-on-edge=%d transit-tangent-parallel=%d other=%d\n",
            degenPhaseCounts[0], degenPhaseCounts[1],
            degenPhaseCounts[2], degenPhaseCounts[3], degenPhaseCounts[4]);
        std::fprintf(stderr,
            "[postproc-extend-dry] distance-hist (relative to L_max): "
            "<0.25=%d <0.5=%d <0.75=%d =L_max=%d\n",
            distBuckets[0], distBuckets[1], distBuckets[2], distBuckets[3]);
        std::fprintf(stderr,
            "[postproc-extend-dry] faces-visited-hist: 1-face=%d 2-face=%d "
            "3-face=%d 4-face=%d 5+=%d\n",
            faceBuckets[0], faceBuckets[1], faceBuckets[2],
            faceBuckets[3], faceBuckets[4]);
        std::fprintf(stderr,
            "[postproc-extend-dry] per-family: fam0-success=%d fam1-success=%d "
            "fam2-success=%d (success = max_length_reached)\n",
            famSuccess[0], famSuccess[1], famSuccess[2]);
        std::fprintf(stderr,
            "[postproc-extend-dry] tangent-transport-hist (face-normal angle, "
            "deg): <1=%d <10=%d <30=%d >=30=%d\n",
            rotBuckets[0], rotBuckets[1], rotBuckets[2], rotBuckets[3]);
        std::fprintf(stderr,
            "[postproc-extend-dry] tangent-drift-hist (|t . n_to| after "
            "transport+nudge): <0.01=%d <0.1=%d <0.5=%d >=0.5=%d\n",
            driftBuckets[0], driftBuckets[1], driftBuckets[2], driftBuckets[3]);

        R.extensionApplied   = true;
        R.extensionMaxLength = maxExt;
        R.extensionPaths     = std::move(paths);
    } // end if (extendLooseEnds)

    (void)L_source;  // only used by the resample pass' log line.
    } // end if (wantPostProc)

    reportProgress(STAGE_DONE, 1, 1, "Done");
    return R;
}

} // namespace wgf
