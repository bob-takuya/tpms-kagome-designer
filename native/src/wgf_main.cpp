// wgf_main.cpp — Emscripten-facing C API for the WGF pipeline.
//
// The JS side builds a base triangle mesh (V, F) and calls wgf_run.
// The pipeline runs end-to-end and returns an opaque handle. JS then
// extracts diagnostics and a flat float array of segment endpoints +
// family labels, and finally frees the handle.

#include "pipeline.h"
#include <memory>
#include <vector>
#include <cstring>

using namespace wgf;

struct WgfHandle {
    PipelineResult result;
    // Flat buffers used by JS to avoid per-segment JS/WASM hops.
    std::vector<double> segmentData;   // size = 6 * nSeg (a.xyz, b.xyz)
    std::vector<int>    segmentFamily; // size = nSeg
    std::vector<int>    segmentFace;   // size = nSeg (base-face index)
};

extern "C" {

// Allocate and run the full pipeline.
//
//   V     : pointer to 3*nV doubles (row-major)
//   nV    : number of vertices
//   F     : pointer to 3*nF int32 (row-major)
//   nF    : number of faces
//   lambdaInit  : Algorithm 1 initial λ (paper: 1000)
//   lambdaMin   : Algorithm 1 minimum λ (paper: 0 → we use 1e-3)
//   alg1MaxIter : Algorithm 1 inner iterations per λ step (paper: ~50)
//   mu          : eq. 7 regularizer (paper: 1e-4)
//   jointIters  : eq. 8 (s,θ) alternations (paper: 10)
//   userScale   : user-facing stripe density multiplier (1 = paper default)
//   useCover    : 1 to run the 6-fold branched cover pipeline, 0 for
//                 single-family base-only mode (diagnostics).
//
// Returns a handle pointer (opaque). The caller must eventually call
// wgf_result_free on it. Returns 0 on failure.
void* wgf_run(const double* V, int nV,
              const int* F, int nF,
              double lambdaInit,
              double lambdaMin,
              int    alg1MaxIter,
              double mu,
              int    jointIters,
              double userScale,
              int    useCover) {
    if (!V || !F || nV < 3 || nF < 1) return nullptr;

    Mesh base;
    base.V.resize(nV, 3);
    for (int i = 0; i < nV; ++i) {
        base.V(i, 0) = V[3*i + 0];
        base.V(i, 1) = V[3*i + 1];
        base.V(i, 2) = V[3*i + 2];
    }
    base.F.resize(nF, 3);
    for (int i = 0; i < nF; ++i) {
        base.F(i, 0) = F[3*i + 0];
        base.F(i, 1) = F[3*i + 1];
        base.F(i, 2) = F[3*i + 2];
    }
    buildHalfEdges(base);

    PipelineOptions opt;
    opt.lambdaInit  = lambdaInit;
    opt.lambdaMin   = lambdaMin;
    opt.alg1MaxIter = alg1MaxIter;
    opt.mu          = mu;
    opt.jointIters  = jointIters;
    opt.userScale   = userScale;
    opt.useCover    = useCover != 0;

    auto* h = new WgfHandle();
    try {
        h->result = runPipeline(base, opt);
    } catch (...) {
        delete h;
        return nullptr;
    }

    // Flatten segment data for JS.
    const auto& segs = h->result.segments;
    h->segmentData.resize(segs.size() * 6);
    h->segmentFamily.resize(segs.size());
    h->segmentFace.resize(segs.size());
    for (size_t i = 0; i < segs.size(); ++i) {
        h->segmentData[6*i + 0] = segs[i].a.x();
        h->segmentData[6*i + 1] = segs[i].a.y();
        h->segmentData[6*i + 2] = segs[i].a.z();
        h->segmentData[6*i + 3] = segs[i].b.x();
        h->segmentData[6*i + 4] = segs[i].b.y();
        h->segmentData[6*i + 5] = segs[i].b.z();
        h->segmentFamily[i] = segs[i].family;
        h->segmentFace[i]   = segs[i].baseFaceIdx;
    }
    return h;
}

int    wgf_result_num_families(void* hPtr) {
    auto* h = (WgfHandle*)hPtr;
    return h ? 3 : 0;
}

int    wgf_result_num_segments(void* hPtr) {
    auto* h = (WgfHandle*)hPtr;
    return h ? (int)h->segmentFamily.size() : 0;
}

// Returns the pointer to the flat doubles buffer (6 * nSeg doubles).
const double* wgf_result_segment_points(void* hPtr) {
    auto* h = (WgfHandle*)hPtr;
    return h ? h->segmentData.data() : nullptr;
}

// Returns the pointer to the family-label buffer (nSeg int).
const int* wgf_result_segment_family(void* hPtr) {
    auto* h = (WgfHandle*)hPtr;
    return h ? h->segmentFamily.data() : nullptr;
}

// Returns the pointer to the base-face-index buffer (nSeg int).
const int* wgf_result_segment_face(void* hPtr) {
    auto* h = (WgfHandle*)hPtr;
    return h ? h->segmentFace.data() : nullptr;
}

double wgf_alg1_curl_initial(void* hPtr) {
    auto* h = (WgfHandle*)hPtr;
    return h ? h->result.alg1BaseInitCurl : 0.0;
}

double wgf_alg1_curl_final(void* hPtr) {
    auto* h = (WgfHandle*)hPtr;
    return h ? h->result.alg1BaseFinalCurl : 0.0;
}

int    wgf_alg1_iters(void* hPtr) {
    auto* h = (WgfHandle*)hPtr;
    return h ? h->result.alg1BaseIters : 0;
}

int    wgf_num_singularities(void* hPtr) {
    auto* h = (WgfHandle*)hPtr;
    return h ? h->result.numSingular : 0;
}

void   wgf_result_free(void* hPtr) {
    delete (WgfHandle*)hPtr;
}

void   wgf_result_free_all(void* hPtr) {
    // Alias kept for symmetry with the exported-function list.
    delete (WgfHandle*)hPtr;
}

} // extern "C"
