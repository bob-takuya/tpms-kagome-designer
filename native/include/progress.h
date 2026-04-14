// progress.h — lightweight progress reporting from C++ to the JS side.
//
// Inside a dedicated Web Worker (where wgf_run actually executes in the
// browser) `self.postMessage({...})` delivers a message to the parent
// main thread immediately, without blocking the worker. This lets the
// long-running pipeline stream phase + item-count progress even though
// wgf_run itself is a single synchronous C call.
//
// Usage from pipeline.h:
//
//   reportProgress(STAGE_ALG1, i, N, "Algorithm 1 sharpening");
//
// In a native (non-Emscripten) build the call compiles to a no-op.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

namespace wgf {

// Stage IDs. Kept small and stable so the TS side can key an explicit
// UI against them (phase label, expected denominator, etc.).
enum Stage {
    STAGE_INIT          = 0,
    STAGE_ROSY          = 1,
    STAGE_ALG1_BASE     = 2,
    STAGE_COVER_BUILD   = 3,
    STAGE_COVER_SPLIT   = 4,
    STAGE_COMPONENT     = 5,   // per-connected-component work
    STAGE_ASSEMBLE      = 6,
    STAGE_DONE          = 7,
};

inline void reportProgress(int stage, int cur, int total, const char* label) {
#ifdef __EMSCRIPTEN__
    // EM_ASM posts directly to the parent thread when the module is
    // running inside a DedicatedWorkerGlobalScope. In the main thread
    // (during unit tests etc.) the guard short-circuits.
    EM_ASM({
        try {
            if (typeof DedicatedWorkerGlobalScope !== 'undefined' &&
                self instanceof DedicatedWorkerGlobalScope) {
                self.postMessage({
                    type:  'wgf-progress',
                    stage: $0,
                    cur:   $1,
                    total: $2,
                    label: UTF8ToString($3)
                });
            }
        } catch (e) { /* ignore */ }
    }, stage, cur, total, label);
#else
    (void)stage; (void)cur; (void)total; (void)label;
#endif
}

} // namespace wgf
