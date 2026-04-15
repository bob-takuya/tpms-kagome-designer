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
#include "permutations.h"
#include "isolines.h"
#include "curl_local_integration.h"
#include "gn_global_integration.h"
#include "paired_rounding.h"
#include "progress.h"
#include <unordered_map>
#include <climits>

namespace wgf {

struct PipelineOptions {
    // Paper-strict defaults (Vekhter et al. 2019 §5.2).
    // No browser-friendly substitutions.
    double lambdaInit     = 1000.0;  // paper §5.2
    double lambdaMin      = 1e-3;    // paper: → 0 (we use 1e-3 to avoid div-by-0)
    int    alg1MaxIter    = 50;      // paper §5.2
    double alg1Tol        = 1e-5;
    double mu             = 1e-4;    // paper eq. 7
    int    jointIters     = 10;      // paper §5.2 (eq. 8 alternations)
    double userScale      = 1.0;
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
    // Number of distinct traced chains (= ribbons before per-family
    // splitting and kagome.ts stitching). Each chain is a single
    // continuous level-set polyline on the cover.
    int    numChains = 0;
    int    numChainsFam[3] = {0, 0, 0};
    // Ribbon length stats (in segments per chain).
    int    chainLenMin = 0;
    int    chainLenMax = 0;
    double chainLenMean = 0;
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
    // Faithful pipeline matching the reference code in
    // https://github.com/the13fools/weaving-geodesic-foliations :
    //
    //   (a) Knoeppel 6-RoSy smooth direction field → w0base
    //       (the paper's Alg 1 on a closed surface with no handles
    //       reduces to the Knoeppel 6-RoSy eigensolve, which we already
    //       computed for step 1 above). We do NOT use R1.w here because
    //       the single-field Alg 1 on a 6-RoSy eigenvector input drifts
    //       individual face angles out of the 60°-canonical range,
    //       breaking the 3-RoSy set alignment across neighbor faces.
    //       The reference's equivalent is a multi-field GaussNewton on
    //       an already-split 3-RoSy state — we don't have that, but
    //       using the unrefined Knoeppel 6-RoSy gives the same 3-RoSy
    //       clean-alignment property.
    //   (b) splitFromRoSy: BFS-consistent expansion into 3 per-face
    //       2D vectors at 60° spacing. This gives a 3-RoSy field on
    //       the base mesh.
    //   (c) reassignAllPermutations: per-edge S₃ × {±1}³ exhaustive
    //       search for the 3×3 signed permutation that best aligns the
    //       two faces' 3-RoSy vectors after parallel transport.
    //   (d) findSingularFieldsPerVertex: per-vertex holonomy walk →
    //       bitmask of singular fields at each vertex.
    //   (e) augmentPs: promote each 3×3 signed permutation to a 6×6
    //       0/1 cover permutation in the block layout (+, −) × field.
    //   (f) buildBranchedCover: BFS-based gluing of 18·|F| corner-layer
    //       nodes → cover-mesh topology. The cover vertices are the
    //       equivalence classes of the gluing graph.
    reportProgress(STAGE_COVER_BUILD, 0, 1, "Cover: 3-RoSy split (Weave::splitFromRosy)");
    Vec wRep3 = splitFromRoSy(base, frames, w0base);

    reportProgress(STAGE_COVER_BUILD, 0, 1, "Cover: per-edge permutations (S3 x {+-1}^3)");
    auto Ps = reassignAllPermutations(base, frames, wRep3);
    {
        PsHistogram H = histogramPs(Ps);
        std::fprintf(stderr,
            "[cover] Ps histogram: identity=%d sign-only=%d permuting=%d / %d edges\n",
            H.identity, H.signOnly, H.permuting, (int)Ps.size());
    }

    reportProgress(STAGE_COVER_BUILD, 0, 1, "Cover: findSingularVertices");
    auto singularFields = findSingularFieldsPerVertex(base, Ps);
    {
        int totSingFields = 0, totSingVerts = 0;
        for (int mask : singularFields) {
            if (mask) {
                totSingVerts++;
                for (int b = 0; b < 3; ++b) if (mask & (1 << b)) totSingFields++;
            }
        }
        std::fprintf(stderr,
            "[cover] singular: %d vertices, %d (vertex, field) singularities / %d verts\n",
            totSingVerts, totSingFields, (int)singularFields.size());
    }

    reportProgress(STAGE_COVER_BUILD, 0, 1, "Cover: augmentPs + BFS glue");
    auto Pcover = augmentPs(Ps);
    auto cov = buildBranchedCover(base, frames, Pcover, singularFields);
    R.coverV = cov.mesh.nV();
    R.coverF = cov.mesh.nF();
    R.numSingular = cov.numSingular;

    // (Per-component direction field init now reads directly from
    // wRep3 so layer assignment matches splitFromRoSy's BFS-shift —
    // phiRefined is no longer needed.)

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

        // (a) Initial direction field on this component.
        //
        // Each cover face carries a layer index ∈ {0..5}, which in the
        // augmented cover layout (permutations.h::augmentPs) means:
        //
        //     field   = lay % 3       (which of the 3 3-RoSy vectors)
        //     signAug = lay < 3 ? +1 : −1
        //
        // So the 2D direction on cover face cf (base face bf, layer k)
        // is `signAug · wRep3[bf, field]`, expressed in bf's 2D frame.
        // We then parallel-transport it into the cover sub-mesh's
        // per-face frame (since sub-meshing may re-orient triangles).
        Vec wS(2 * S.mesh.nF());
        for (int f = 0; f < S.mesh.nF(); ++f) {
            int covFace = S.oldFaceIdx[f];
            int bf  = cov.coverFaceBaseF[covFace];
            int lay = cov.coverFaceLayer[covFace];
            int field = lay % 3;
            double sign = (lay < 3 ? 1.0 : -1.0);
            Eigen::Vector2d v2 = sign * repVec(wRep3, bf, field);
            Eigen::Vector3d d3 =
                v2.x() * frames[bf].e1 +
                v2.y() * frames[bf].e2;
            double u = d3.dot(Sframes[f].e1);
            double v = d3.dot(Sframes[f].e2);
            double Ln = std::sqrt(u*u + v*v);
            if (Ln < 1e-12) { u = 1; v = 0; Ln = 1; }
            wS[2*f]     = u / Ln;
            wS[2*f + 1] = v / Ln;
        }

        // (b) Per-component Alg 1 — paper §5.2 schedule.
        Alg1Result RS = runAlg1(S.mesh, Sframes, wS, {},
                                /*lambdaInit=*/opt.lambdaInit,
                                /*lambdaMin =*/opt.lambdaMin,
                                /*maxIter   =*/opt.alg1MaxIter,
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
        // Paper-strict: WeaveHook.h:109-110 in the reference sets
        // globalAlternations=10, globalPowerIters=10.
        Vec thetaComp;
        gnGlobalIntegration(S.mesh, Sframes, RS.w, sCompRaw, thetaComp,
                            /*outerIters=*/10,
                            /*powerIters=*/10,
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
        // Diagnostic: theta range on the cover before and after paired
        // rounding. A smooth angular theta should span roughly [−π, π].
        auto thetaStats = [&](const char* tag) {
            double tmn = 1e300, tmx = -1e300, tabssum = 0;
            int nV = (int)thetaCover.size();
            for (int i = 0; i < nV; ++i) {
                double v = thetaCover[i];
                if (v < tmn) tmn = v;
                if (v > tmx) tmx = v;
                tabssum += std::fabs(v);
            }
            std::fprintf(stderr,
                "[cover] theta %s paired-rounding: [%.3f, %.3f]  |theta|mean=%.3f  nV=%d\n",
                tag, tmn, tmx, tabssum / std::max(1, nV), nV);
        };
        thetaStats("before");
        pairedRoundTheta(cov.mesh, cov, thetaCover, numISOLines);
        thetaStats("after ");

        // 7. Extract angular isolines on the cover. The tracer produces
        // chains (groups of segments with shared chainId) that follow a
        // level set from face to face across the cover.
        reportProgress(STAGE_ASSEMBLE, 0, 1, "Extracting isolines");
        auto segs = extractAngularIsolines(cov.mesh, thetaCover, numISOLines);
        {
            int perLevel[32] = {0};
            for (const auto& s : segs) if (s.iso < 32) perLevel[s.iso]++;
            int lim = std::min(numISOLines, 32);
            std::fprintf(stderr, "[cover] segs per isoline level:");
            for (int i = 0; i < lim; ++i) std::fprintf(stderr, " %d", perLevel[i]);
            std::fprintf(stderr, "  (total %d)\n", (int)segs.size());
        }

        // 8. Project each cover segment back to the base.
        //
        // With the faithful 3-RoSy cover (Vekhter §5.1) most cover edges
        // have Ps = ±I, i.e. the per-edge permutation does not permute
        // field indices — only potentially flips signs. Signs live in
        // the "upper half" (layers 3-5) of the cover, so walking a
        // level set across such edges either keeps the field index
        // constant or jumps by exactly 3 (which has the same field mod
        // 3). Therefore a traced chain's `family = layer % 3` is
        // constant almost everywhere, yielding continuous ribbons.
        //
        // Per-segment family is just (cover layer) mod 3.
        //
        // We also track the chain (= ribbon) statistics: count of
        // distinct chainIds overall and per family, plus
        // min/max/mean chain length in segments. These numbers are
        // what "how many ribbons?" really means — the raw segment
        // count is ~O(#cover-faces × numISOLines) and is much larger.
        std::unordered_map<int, int>  chainLen;
        std::unordered_map<int, int>  chainFam;  // fam of first-seen segment
        for (const auto& s : segs) {
            int cf = s.faceIdx;
            int bf = cov.coverFaceBaseF[cf];
            int lay = cov.coverFaceLayer[cf];
            int fam = lay % 3;
            ProjectedSegment p;
            p.a = s.a;
            p.b = s.b;
            p.baseFaceIdx = bf;
            p.family = fam;
            R.segments.push_back(p);
            R.numSegmentsFam[fam]++;

            auto it = chainLen.find(s.chainId);
            if (it == chainLen.end()) {
                chainLen[s.chainId] = 1;
                chainFam[s.chainId] = fam;
            } else {
                it->second++;
            }
        }
        R.numChains = (int)chainLen.size();
        if (R.numChains > 0) {
            int minL = INT_MAX, maxL = 0;
            long long sum = 0;
            for (const auto& kv : chainLen) {
                int len = kv.second;
                sum += len;
                if (len < minL) minL = len;
                if (len > maxL) maxL = len;
            }
            R.chainLenMin = minL;
            R.chainLenMax = maxL;
            R.chainLenMean = (double)sum / R.numChains;
        }
        for (const auto& kv : chainFam) {
            R.numChainsFam[kv.second]++;
        }

        // Chain length histogram (log-bucketed): 1, 2, 3-4, 5-8, 9-16,
        // 17-32, 33-64, 65+. Useful to diagnose whether ribbons are
        // genuinely long or just single-face fragments.
        {
            int bins[8] = {0};
            for (const auto& kv : chainLen) {
                int L = kv.second;
                int b = 0;
                if      (L <= 1)  b = 0;
                else if (L == 2)  b = 1;
                else if (L <= 4)  b = 2;
                else if (L <= 8)  b = 3;
                else if (L <= 16) b = 4;
                else if (L <= 32) b = 5;
                else if (L <= 64) b = 6;
                else              b = 7;
                bins[b]++;
            }
            std::fprintf(stderr,
                "[cover] chain length hist: 1=%d 2=%d 3-4=%d 5-8=%d "
                "9-16=%d 17-32=%d 33-64=%d 65+=%d   total=%d\n",
                bins[0], bins[1], bins[2], bins[3],
                bins[4], bins[5], bins[6], bins[7], R.numChains);
        }
    }

    reportProgress(STAGE_DONE, 1, 1, "Done");
    return R;
}

} // namespace wgf
