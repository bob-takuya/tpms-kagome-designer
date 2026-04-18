// wgf_cli.cpp — native CLI wrapper around the Weaving Geodesic
// Foliations pipeline. Intended for running in Google Colab (or any
// native Linux/macOS environment) to offload the heavy computation
// from the in-browser WASM build.
//
// Input (stdin) — plain text, line oriented:
//
//     # wgf-input v1
//     V <nV>
//     <x> <y> <z>
//     ...
//     F <nF>
//     <a> <b> <c>
//     ...
//     OPTS lambdaInit=1000 lambdaMin=1e-3 alg1MaxIter=50 mu=1e-4  (etc.)
//     END
//
// Output (stdout) — same style:
//
//     # wgf-output v2
//     META initialCurl=<v> finalCurl=<v> iterations=<v> numSingular=<v>  (etc.)
//     SEG <nSeg>
//     <ax> <ay> <az> <bx> <by> <bz> <family> <baseFaceIdx> <chainId>
//     ...
//     END
//
// v2 note: `chainId` (9th column, >= 0) groups segments that form a
// single contiguous polyline on the cover/base. -1 means "unknown"
// (e.g. the useCover=false debug path). Readers of the v1 8-column
// format should treat a missing chainId as -1 for backward compat.
//
// Diagnostic progress is written to stderr; stdout is just the result
// so that `./wgf_cli < input.txt > output.txt` works unmodified.

#include "pipeline.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

using namespace wgf;

namespace {

struct CliInput {
    int nV = 0, nF = 0;
    std::vector<double> Vbuf;
    std::vector<int>    Fbuf;
    PipelineOptions     opt;
};

bool parseInput(std::istream& in, CliInput& out) {
    // Paper-strict defaults for the CLI (the user is paying for a native
    // run, so give them the full-quality settings out of the box).
    out.opt.lambdaInit  = 1000.0;
    out.opt.lambdaMin   = 1e-3;
    out.opt.alg1MaxIter = 50;
    out.opt.mu          = 1e-4;
    out.opt.jointIters  = 10;
    out.opt.userScale   = 1.0;
    out.opt.useCover    = true;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string tag;
        is >> tag;
        if (tag == "V") {
            is >> out.nV;
            out.Vbuf.resize((size_t)3 * out.nV);
            for (int i = 0; i < out.nV; ++i) {
                in >> out.Vbuf[3*i] >> out.Vbuf[3*i + 1] >> out.Vbuf[3*i + 2];
            }
            std::getline(in, line);
        } else if (tag == "F") {
            is >> out.nF;
            out.Fbuf.resize((size_t)3 * out.nF);
            for (int i = 0; i < out.nF; ++i) {
                in >> out.Fbuf[3*i] >> out.Fbuf[3*i + 1] >> out.Fbuf[3*i + 2];
            }
            std::getline(in, line);
        } else if (tag == "OPTS") {
            std::string kv;
            while (is >> kv) {
                auto eq = kv.find('=');
                if (eq == std::string::npos) continue;
                std::string key = kv.substr(0, eq);
                std::string val = kv.substr(eq + 1);
                try {
                    if      (key == "lambdaInit")  out.opt.lambdaInit  = std::stod(val);
                    else if (key == "lambdaMin")   out.opt.lambdaMin   = std::stod(val);
                    else if (key == "alg1MaxIter") out.opt.alg1MaxIter = std::stoi(val);
                    else if (key == "alg1Tol")     out.opt.alg1Tol     = std::stod(val);
                    else if (key == "mu")          out.opt.mu          = std::stod(val);
                    else if (key == "jointIters")  out.opt.jointIters  = std::stoi(val);
                    else if (key == "userScale")   out.opt.userScale   = std::stod(val);
                    else if (key == "useCover")    out.opt.useCover    = (std::stoi(val) != 0);
                    else if (key == "diagOnly")    out.opt.diagOnly    = std::stoi(val);
                    else if (key == "maxVerts")    out.opt.maxVerts    = std::stoi(val);
                } catch (...) {
                    std::fprintf(stderr, "[wgf-cli] malformed option: %s\n", kv.c_str());
                }
            }
        } else if (tag == "END") {
            break;
        }
    }
    return out.nV > 0 && out.nF > 0;
}

} // anonymous namespace

int main(int /*argc*/, char** /*argv*/) {
    CliInput cin_;
    if (!parseInput(std::cin, cin_)) {
        std::fprintf(stderr, "[wgf-cli] failed to parse input (expected V / F / OPTS / END)\n");
        return 1;
    }

    // Environment-variable override for diag-only fast mode (handy from
    // Colab without editing wgf-input.txt).
    if (const char* env = std::getenv("WGF_DIAG_ONLY")) {
        int v = std::atoi(env);
        if (v > 0) cin_.opt.diagOnly = v;
    }

    Mesh mesh;
    mesh.V.resize(cin_.nV, 3);
    for (int i = 0; i < cin_.nV; ++i) {
        mesh.V(i, 0) = cin_.Vbuf[3*i];
        mesh.V(i, 1) = cin_.Vbuf[3*i + 1];
        mesh.V(i, 2) = cin_.Vbuf[3*i + 2];
    }
    mesh.F.resize(cin_.nF, 3);
    for (int i = 0; i < cin_.nF; ++i) {
        mesh.F(i, 0) = cin_.Fbuf[3*i];
        mesh.F(i, 1) = cin_.Fbuf[3*i + 1];
        mesh.F(i, 2) = cin_.Fbuf[3*i + 2];
    }
    buildHalfEdges(mesh);

    std::fprintf(stderr, "[wgf-cli] mesh  : V=%d F=%d E_int=%d\n",
                 mesh.nV(), mesh.nF(), mesh.nE());
    std::fprintf(stderr, "[wgf-cli] opts  : lambdaInit=%.3g lambdaMin=%.3g alg1MaxIter=%d "
                 "mu=%.3g jointIters=%d userScale=%.3g useCover=%d diagOnly=%d maxVerts=%d\n",
                 cin_.opt.lambdaInit, cin_.opt.lambdaMin, cin_.opt.alg1MaxIter,
                 cin_.opt.mu, cin_.opt.jointIters, cin_.opt.userScale,
                 cin_.opt.useCover ? 1 : 0, cin_.opt.diagOnly, cin_.opt.maxVerts);

    PipelineResult R;
    try {
        R = runPipeline(mesh, cin_.opt);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[wgf-cli] FAILED: %s\n", e.what());
        return 2;
    }

    std::fprintf(stderr, "[wgf-cli] result: segments=%zu components=%d singular=%d "
                 "curl %.3e -> %.3e (%.1f%% reduction) base Alg1 iters=%d\n",
                 R.segments.size(), R.numComponents, R.numSingular,
                 R.alg1BaseInitCurl, R.alg1BaseFinalCurl,
                 (1.0 - R.alg1BaseFinalCurl / std::max(R.alg1BaseInitCurl, 1e-30)) * 100.0,
                 R.alg1BaseIters);
    std::fprintf(stderr, "[wgf-cli] family counts: %d / %d / %d\n",
                 R.numSegmentsFam[0], R.numSegmentsFam[1], R.numSegmentsFam[2]);

    // Write output on stdout.
    std::cout << "# wgf-output v2\n";
    std::cout << "META"
              << " initialCurl="  << R.alg1BaseInitCurl
              << " finalCurl="    << R.alg1BaseFinalCurl
              << " iterations="   << R.alg1BaseIters
              << " numSingular="  << R.numSingular
              << " segments="     << R.segments.size()
              << " components="   << R.numComponents
              << " fam0="         << R.numSegmentsFam[0]
              << " fam1="         << R.numSegmentsFam[1]
              << " fam2="         << R.numSegmentsFam[2]
              << "\n";
    std::cout << "SEG " << R.segments.size() << "\n";
    // Use printf for the SEG rows so the column format matches the
    // wgf-output v2 spec exactly (%.9f for positions). stdio/iostream
    // sync is on by default, so intermixing with std::cout is safe.
    for (const auto& s : R.segments) {
        std::printf("%.9f %.9f %.9f %.9f %.9f %.9f %d %d %d\n",
                    s.a.x(), s.a.y(), s.a.z(),
                    s.b.x(), s.b.y(), s.b.z(),
                    s.family, s.baseFaceIdx, s.chainId);
    }
    std::cout << "END\n";

    return 0;
}
