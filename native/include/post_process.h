// post_process.h — isoline post-processing (Vekhter 2019 §5.2).
//
// This header implements the first post-processing step from §5.2:
//
//   "resample the isolines so that all segments are approximately the
//    same length"
//
// The subsequent steps (cut at sharp turns, prune short chains, extend
// near-coincident endpoints) are intentionally NOT implemented here;
// they will follow in their own PRs.
//
// Pipeline position: runs AFTER the centerline extraction emits its
// ProjectedSegments (one per traced isoline face-crossing) and BEFORE
// the pipeline returns. Takes raw ProjectedSegments in, returns
// resampled ProjectedSegments with near-uniform length.
//
// Algorithm:
//   A. Bucket segments by chainId and rebuild ordered polylines from
//      endpoint adjacency (PR #24 made endpoints bit-identical across
//      shared cover edges, so adjacency keys line up exactly).
//      A single chainId may decompose into multiple disjoint polylines;
//      each becomes its own Polyline with the shared chainId.
//   B. For each polyline, resample at arclength-uniform step L_target.
//      Open chains keep their two endpoints fixed (n samples → n−1 output
//      segments); closed loops use n samples spaced uniformly around the
//      perimeter (n output segments with a wrap-around closing edge).
//   C. Emit one ProjectedSegment per output segment, keeping chainId and
//      family, and inheriting baseFaceIdx from the source input interval.

#pragma once

#include "isolines.h"
#include "mesh.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace wgf {

// ---------------------------------------------------------------------------
// Polyline: ordered-sample representation of a single traced isoline.
// ---------------------------------------------------------------------------
//
// Indexing:
//   points.size()  == N
//   faceIds.size() == N - 1   (one per interval [points[i-1], points[i]])
//
// For closed loops we store the last-first duplicate explicitly (i.e.
// points[N-1] ≈ points[0]), so that faceIds covers the wrap-around
// interval and the resampler's cumulative-arclength pass sees the full
// perimeter as a monotone scan.
struct Polyline {
    int chainId = -1;
    int family  = 0;
    std::vector<Eigen::Vector3d> points;
    std::vector<int> faceIds;
    bool isClosed = false;
};

// Per-polyline arclength record used by the chain-level diagnostics in
// pipeline.h. Captures how much each reconstructed polyline shrank (or
// grew) through the resample pass so that open chains, closed loops, and
// individual worst offenders can be reported separately.
struct ResamplePolyStat {
    int    chainId   = -1;
    int    family    = 0;
    bool   isClosed  = false;
    // True when the short-chain fast-path preserved this polyline verbatim
    // instead of running the arclength-uniform resampler. Used by the
    // fast-path-only diff histogram in pipeline.h to verify that preserved
    // chains keep beforeLen == afterLen to numerical precision.
    bool   fastPath  = false;
    double beforeLen = 0.0;
    double afterLen  = 0.0;
};

struct ResampleStats {
    double targetLength   = 0.0;
    double meshMeanEdge   = 0.0;
    int    inputChains    = 0;
    int    outputPolylines = 0;
    int    openChains     = 0;
    int    closedLoops    = 0;
    int    inputSegs      = 0;
    int    outputSegs     = 0;
    double inputSegLenMean = 0, inputSegLenStd = 0;
    double inputSegLenMin  = 0, inputSegLenMax = 0;
    double outputSegLenMean = 0, outputSegLenStd = 0;
    double outputSegLenMin  = 0, outputSegLenMax = 0;
    double inputTotalArclen  = 0;
    double outputTotalArclen = 0;

    // Short-chain fast-path counters (E-1.5). A chain is "fast-path" when
    // it was too short relative to L_target to resample without collapsing
    // arclength, so the input was preserved verbatim (see
    // resampleOpenChain / resampleClosedLoop).
    std::size_t openFastPath    = 0;
    std::size_t closedFastPath  = 0;
    std::size_t openResampled   = 0;
    std::size_t closedResampled = 0;

    // Per-polyline before/after arclength (one entry per reconstructed
    // Polyline, including closed loops). Populated by
    // resampleProjectedSegments and consumed by the chain-level diagnostic
    // logger in pipeline.h.
    std::vector<ResamplePolyStat> perPoly;
};

// Mean length of all undirected edges (including boundary edges) of the
// base mesh. Used as the default L_target when no explicit value is
// supplied on the CLI.
inline double meshMeanEdgeLength(const Mesh& m) {
    // Every directed half-edge counts each undirected edge twice (once
    // per side) except boundary edges (once). Summing the per-half-edge
    // length and dividing by the half-edge count gives the mean over
    // physical sides, which is within a single-boundary correction of
    // the true undirected-edge mean — close enough for a default scale.
    if (m.he.empty()) {
        // Fallback: derive from F directly if half-edges haven't been
        // built yet.
        double sum = 0.0;
        int count  = 0;
        for (int f = 0; f < m.nF(); ++f) {
            for (int e = 0; e < 3; ++e) {
                int a = m.F(f, e);
                int b = m.F(f, (e + 1) % 3);
                sum += (m.V.row(a) - m.V.row(b)).norm();
                ++count;
            }
        }
        return (count > 0) ? (sum / count) : 0.0;
    }
    double sum = 0.0;
    int count  = 0;
    for (int h = 0; h < (int)m.he.size(); ++h) {
        int vEnd   = m.he[h].vertex;
        int vStart = m.he[m.he[h].prev].vertex;
        sum += (m.V.row(vEnd) - m.V.row(vStart)).norm();
        ++count;
    }
    return (count > 0) ? (sum / count) : 0.0;
}

namespace detail {

struct QKey {
    std::int64_t x, y, z;
    bool operator==(const QKey& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

struct QKeyHash {
    std::size_t operator()(const QKey& k) const noexcept {
        // Splittable FNV-ish mixer; sufficient for ~10^6 entries.
        std::uint64_t h = 1469598103934665603ULL;
        h ^= (std::uint64_t)k.x; h *= 1099511628211ULL;
        h ^= (std::uint64_t)k.y; h *= 1099511628211ULL;
        h ^= (std::uint64_t)k.z; h *= 1099511628211ULL;
        return (std::size_t)h;
    }
};

inline QKey quantKey(const Eigen::Vector3d& p) {
    // eps=1e-9 matches the bit-identical-endpoint invariant established
    // by PR #24: shared endpoints across adjacent cover segments are
    // produced by the same linear-interpolation formula on the same
    // edge with the same t, so they agree to well under 1e-9 relative.
    constexpr double eps = 1e-9;
    return QKey{
        (std::int64_t)std::llround(p.x() / eps),
        (std::int64_t)std::llround(p.y() / eps),
        (std::int64_t)std::llround(p.z() / eps)
    };
}

} // namespace detail

// ---------------------------------------------------------------------------
// Step A: per-chain ordered polyline reconstruction.
// ---------------------------------------------------------------------------
//
// Groups ProjectedSegments by chainId and, within each group, walks
// endpoint adjacency to produce one or more Polylines. Disjoint
// sub-components inside a single chainId (can happen if the tracer
// emitted two unconnected arcs under the same id) are returned as
// separate Polylines with the shared chainId; no input segment is
// duplicated or dropped.
inline std::vector<Polyline> buildOrderedPolylines(
    const std::vector<ProjectedSegment>& segs)
{
    using namespace detail;

    // Bucket by chainId. Segments with chainId < 0 (e.g. the
    // useCover=false debug path) each become their own single-segment
    // polyline to keep the rest of the pipeline schema-uniform.
    std::unordered_map<int, std::vector<int>> byChain;
    byChain.reserve(segs.size() / 8 + 1);
    std::vector<int> orphans;
    for (int i = 0; i < (int)segs.size(); ++i) {
        int cid = segs[i].chainId;
        if (cid < 0) orphans.push_back(i);
        else         byChain[cid].push_back(i);
    }

    std::vector<Polyline> out;
    out.reserve(byChain.size() + orphans.size());

    for (auto& kv : byChain) {
        const std::vector<int>& bucket = kv.second;
        if (bucket.empty()) continue;

        // Local index inside bucket; avoids a global visited[] vector.
        std::unordered_map<int, int> bucketIdx;
        bucketIdx.reserve(bucket.size() * 2);
        for (int k = 0; k < (int)bucket.size(); ++k) bucketIdx[bucket[k]] = k;

        // Adjacency: endpoint -> list of (segIdx, side) where side 0 = a, 1 = b.
        std::unordered_map<QKey, std::vector<std::pair<int,int>>, QKeyHash> adj;
        adj.reserve(bucket.size() * 2);
        for (int gi : bucket) {
            adj[quantKey(segs[gi].a)].push_back({gi, 0});
            adj[quantKey(segs[gi].b)].push_back({gi, 1});
        }

        std::vector<char> visited(bucket.size(), 0);

        auto walkFrom = [&](int startGi, int startSide, bool forceClosed) {
            Polyline pl;
            pl.chainId = segs[startGi].chainId;
            pl.family  = segs[startGi].family;
            pl.isClosed = forceClosed;

            const Eigen::Vector3d startPt = (startSide == 0)
                ? segs[startGi].a : segs[startGi].b;
            const Eigen::Vector3d otherPt = (startSide == 0)
                ? segs[startGi].b : segs[startGi].a;

            pl.points.push_back(startPt);
            pl.points.push_back(otherPt);
            pl.faceIds.push_back(segs[startGi].baseFaceIdx);
            visited[bucketIdx[startGi]] = 1;

            Eigen::Vector3d cur = otherPt;
            const QKey startKey = quantKey(startPt);
            while (true) {
                if (forceClosed && quantKey(cur) == startKey) break;
                auto it = adj.find(quantKey(cur));
                if (it == adj.end()) break;
                int nextGi = -1, nextSide = -1;
                for (auto& ps : it->second) {
                    int kk = bucketIdx[ps.first];
                    if (!visited[kk]) { nextGi = ps.first; nextSide = ps.second; break; }
                }
                if (nextGi < 0) break;
                visited[bucketIdx[nextGi]] = 1;
                Eigen::Vector3d nextOther = (nextSide == 0)
                    ? segs[nextGi].b : segs[nextGi].a;
                pl.points.push_back(nextOther);
                pl.faceIds.push_back(segs[nextGi].baseFaceIdx);
                cur = nextOther;
            }

            // If the walk returned to the start endpoint naturally
            // (without the forceClosed test), classify as closed.
            if (!forceClosed && pl.points.size() >= 3 &&
                quantKey(pl.points.back()) == quantKey(pl.points.front())) {
                pl.isClosed = true;
            }
            return pl;
        };

        // 1) Open chains: start from each degree-1 endpoint.
        for (auto& ae : adj) {
            if (ae.second.size() != 1) continue;
            int gi   = ae.second[0].first;
            int side = ae.second[0].second;
            if (visited[bucketIdx[gi]]) continue;
            Polyline pl = walkFrom(gi, side, /*forceClosed=*/false);
            out.push_back(std::move(pl));
        }

        // 2) Closed loops: any segment not yet visited is part of a
        //    closed component (no degree-1 endpoints remain touching it).
        for (int k = 0; k < (int)bucket.size(); ++k) {
            if (visited[k]) continue;
            int gi = bucket[k];
            Polyline pl = walkFrom(gi, /*startSide=*/0, /*forceClosed=*/true);
            out.push_back(std::move(pl));
        }
    }

    // Orphan segments (chainId < 0): emit each as a trivial 2-point
    // open polyline. Preserves count parity for the downstream stats.
    for (int gi : orphans) {
        Polyline pl;
        pl.chainId = segs[gi].chainId;
        pl.family  = segs[gi].family;
        pl.isClosed = false;
        pl.points.push_back(segs[gi].a);
        pl.points.push_back(segs[gi].b);
        pl.faceIds.push_back(segs[gi].baseFaceIdx);
        out.push_back(std::move(pl));
    }

    return out;
}

// ---------------------------------------------------------------------------
// Step B: near-uniform resample.
// ---------------------------------------------------------------------------

// Open-chain tuning constants (E-1.6). Bumped from 1.5/3 to 2.5/5 so that
// medium-shrink open chains (L/target ~ 1.5-2.5, n=3 -> 30-62% shrink in
// production data) take the fast-path or get enough samples (>=4 chords)
// to follow curvature. Kept as named constants so future tuning lives in
// one place. Closed-loop constants stay inline at their call sites because
// the closed path is already within tolerance and we want changes here to
// not bleed across.
constexpr double OPEN_FASTPATH_MULT = 2.5;
constexpr int    OPEN_MIN_N         = 5;

// Returns true when the short-chain fast-path kicked in (input preserved
// verbatim). Caller uses this to count openFastPath vs openResampled.
inline bool resampleOpenChain(
    const std::vector<Eigen::Vector3d>& input,
    const std::vector<int>&             inFaceIds,
    double                              L_target,
    std::vector<Eigen::Vector3d>&       output,
    std::vector<int>&                   outFaceIds,
    bool                                disableFastPath = false)
{
    output.clear();
    outFaceIds.clear();
    if (input.size() < 2) {
        output = input;
        outFaceIds = inFaceIds;
        return true;
    }

    std::vector<double> s(input.size(), 0.0);
    for (std::size_t i = 1; i < input.size(); ++i) {
        s[i] = s[i-1] + (input[i] - input[i-1]).norm();
    }
    const double totalLen = s.back();
    if (totalLen <= 0.0) {
        output.push_back(input.front());
        output.push_back(input.back());
        outFaceIds.push_back(inFaceIds.empty() ? -1 : inFaceIds.front());
        return true;
    }

    // Short-chain fast-path (E-1.5, widened in E-1.6): when the total
    // arclength is shorter than OPEN_FASTPATH_MULT * target, resampling
    // produces too few sample points to follow curvature and the chord
    // sum collapses below the input arclength. Production data after
    // PR #27 still showed open chains with L/target in the 1.5-2.9 range
    // shrinking 30-62%, all from the n=3 (2-chord) regime. Preserve the
    // input verbatim instead. PR #24 / #25 already guarantee 3D endpoint
    // continuity across adjacent cover segments, so emitting the input
    // points directly keeps arclength and shape exactly.
    if (!disableFastPath && totalLen < L_target * OPEN_FASTPATH_MULT) {
        output = input;
        outFaceIds = inFaceIds;
        return true;
    }

    // ceil instead of round biases n upward when L/target has a fractional
    // part, shortening each segment below target and making the chord sum
    // track arclength more faithfully. The "+1" converts segments to sample
    // points; OPEN_MIN_N guarantees at least 4 output segments (5 points)
    // for chains that just clear the fast-path cutoff, where n=3 would
    // otherwise reproduce the medium-shrink regime we just widened past.
    int n = std::max(OPEN_MIN_N, (int)std::ceil(totalLen / std::max(L_target, 1e-30)) + 1);
    output.reserve(n);
    outFaceIds.reserve(n);

    for (int k = 0; k < n; ++k) {
        double t = (totalLen * k) / (n - 1);
        auto it = std::upper_bound(s.begin(), s.end(), t);
        std::size_t i = (it == s.begin()) ? 1 : (std::size_t)std::distance(s.begin(), it);
        if (i >= input.size()) i = input.size() - 1;
        double segLen = s[i] - s[i-1];
        double alpha = (segLen > 1e-12) ? (t - s[i-1]) / segLen : 0.0;
        output.push_back((1.0 - alpha) * input[i-1] + alpha * input[i]);
        int face = (inFaceIds.size() >= i) ? inFaceIds[i-1] : -1;
        outFaceIds.push_back(face);
    }
    return false;
}

// Returns true when the short-chain fast-path kicked in.
inline bool resampleClosedLoop(
    const std::vector<Eigen::Vector3d>& input,
    const std::vector<int>&             inFaceIds,
    double                              L_target,
    std::vector<Eigen::Vector3d>&       output,
    std::vector<int>&                   outFaceIds,
    bool                                disableFastPath = false)
{
    output.clear();
    outFaceIds.clear();
    if (input.size() < 3) {
        output = input;
        outFaceIds = inFaceIds;
        return true;
    }

    std::vector<double> s(input.size(), 0.0);
    for (std::size_t i = 1; i < input.size(); ++i) {
        s[i] = s[i-1] + (input[i] - input[i-1]).norm();
    }
    const double totalLen = s.back();
    if (totalLen <= 0.0) {
        output.push_back(input.front());
        outFaceIds.push_back(inFaceIds.empty() ? -1 : inFaceIds.front());
        return true;
    }

    // Short-chain fast-path (E-1.5): closed loops shorter than ~3 * target
    // would resample to n=3 (a triangle) and lose perimeter arclength to
    // the three chord approximations. Preserve the input verbatim instead.
    //
    // Convention translation: buildOrderedPolylines stores closed loops
    // with the start vertex duplicated at the end (N+1 points, N face ids,
    // case A). polylinesToSegments expects N unique points and closes the
    // loop via (k+1)%N. Drop the trailing duplicate so the output obeys
    // the resampled-closed convention.
    if (!disableFastPath && totalLen < L_target * 3.0) {
        output.assign(input.begin(), input.end() - 1);
        outFaceIds = inFaceIds;
        return true;
    }

    // ceil biases n upward (shorter segments, better arclength tracking);
    // min n = 6 guarantees at least a hexagonal approximation and avoids
    // the triangle degeneracy that n=3 would otherwise produce for loops
    // just above the fast-path cutoff.
    int n = std::max(6, (int)std::ceil(totalLen / std::max(L_target, 1e-30)));
    output.reserve(n);
    outFaceIds.reserve(n);

    for (int k = 0; k < n; ++k) {
        // Note the denominator: n, not n-1. This is the defining
        // difference between open (endpoints fixed) and closed (samples
        // spaced uniformly around the perimeter, no endpoint repetition).
        double t = (totalLen * k) / n;
        auto it = std::upper_bound(s.begin(), s.end(), t);
        std::size_t i = (it == s.begin()) ? 1 : (std::size_t)std::distance(s.begin(), it);
        if (i >= input.size()) i = input.size() - 1;
        double segLen = s[i] - s[i-1];
        double alpha = (segLen > 1e-12) ? (t - s[i-1]) / segLen : 0.0;
        output.push_back((1.0 - alpha) * input[i-1] + alpha * input[i]);
        int face = (inFaceIds.size() >= i) ? inFaceIds[i-1] : -1;
        outFaceIds.push_back(face);
    }
    return false;
}

// ---------------------------------------------------------------------------
// Step B (wrapper): resample a single Polyline.
// ---------------------------------------------------------------------------
//
// Returns true when the short-chain fast-path preserved the input verbatim.
// Callers use this to update the per-polyline and aggregate fast-path stats.
inline bool resamplePolyline(const Polyline& in, double L_target, Polyline& out,
                             bool disableFastPath = false) {
    out.chainId  = in.chainId;
    out.family   = in.family;
    out.isClosed = in.isClosed;

    bool fastPath;
    if (in.isClosed) {
        fastPath = resampleClosedLoop(in.points, in.faceIds, L_target,
                                      out.points, out.faceIds, disableFastPath);
    } else {
        fastPath = resampleOpenChain(in.points, in.faceIds, L_target,
                                     out.points, out.faceIds, disableFastPath);
    }
    return fastPath;
}

// ---------------------------------------------------------------------------
// Step C: resampled polyline → ProjectedSegment list.
// ---------------------------------------------------------------------------
inline std::vector<ProjectedSegment> polylinesToSegments(
    const std::vector<Polyline>& polys)
{
    std::vector<ProjectedSegment> out;
    std::size_t reserveHint = 0;
    for (const auto& pl : polys) {
        if (pl.points.size() < 2) continue;
        reserveHint += pl.isClosed ? pl.points.size()
                                   : (pl.points.size() - 1);
    }
    out.reserve(reserveHint);

    for (const auto& pl : polys) {
        const int N = (int)pl.points.size();
        if (N < 2) continue;

        if (pl.isClosed) {
            for (int k = 0; k < N; ++k) {
                ProjectedSegment p;
                p.a = pl.points[k];
                p.b = pl.points[(k + 1) % N];
                p.chainId = pl.chainId;
                p.family  = pl.family;
                p.baseFaceIdx = (k < (int)pl.faceIds.size()) ? pl.faceIds[k] : -1;
                out.push_back(p);
            }
        } else {
            for (int k = 0; k < N - 1; ++k) {
                ProjectedSegment p;
                p.a = pl.points[k];
                p.b = pl.points[k + 1];
                p.chainId = pl.chainId;
                p.family  = pl.family;
                p.baseFaceIdx = (k < (int)pl.faceIds.size()) ? pl.faceIds[k] : -1;
                out.push_back(p);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Combined helper + stats.
// ---------------------------------------------------------------------------
namespace detail {
inline void segmentLengthStats(
    const std::vector<ProjectedSegment>& segs,
    double& mean, double& stddev, double& mn, double& mx, double& total)
{
    mean = stddev = total = 0.0;
    mn = std::numeric_limits<double>::infinity();
    mx = 0.0;
    if (segs.empty()) { mn = 0.0; return; }
    for (const auto& s : segs) {
        double L = (s.b - s.a).norm();
        total += L;
        if (L < mn) mn = L;
        if (L > mx) mx = L;
    }
    mean = total / (double)segs.size();
    double acc = 0.0;
    for (const auto& s : segs) {
        double L = (s.b - s.a).norm();
        double d = L - mean;
        acc += d * d;
    }
    stddev = std::sqrt(acc / (double)segs.size());
}
} // namespace detail

inline std::vector<ProjectedSegment> resampleProjectedSegments(
    const std::vector<ProjectedSegment>& in,
    double                               L_target,
    ResampleStats&                       stats,
    bool                                 disableFastPath = false)
{
    stats = ResampleStats{};
    stats.targetLength = L_target;
    stats.inputSegs = (int)in.size();

    // Count distinct chainIds in input (orphans with cid<0 each count once).
    {
        std::unordered_map<int, char> seen;
        int orphans = 0;
        for (const auto& s : in) {
            if (s.chainId < 0) ++orphans;
            else seen[s.chainId] = 1;
        }
        stats.inputChains = (int)seen.size() + orphans;
    }

    auto polys = buildOrderedPolylines(in);

    // Arclength that matches what polylinesToSegments actually emits.
    //
    // Input polylines (buildOrderedPolylines): open chains store N points
    // and N-1 segments; closed loops store N+1 points with the start
    // duplicated at the end, so the wrap edge is already in the consecutive
    // differences.
    //
    // Resampled polylines: open chains store N points and N-1 segments (same
    // convention); closed loops store N points and polylinesToSegments adds
    // the wrap edge (last -> first) implicitly, so we must include it here
    // to compare apples-to-apples with the input's total.
    auto polylineArclen = [](const Polyline& pl, bool addWrap) -> double {
        double L = 0.0;
        const int N = (int)pl.points.size();
        if (N < 2) return 0.0;
        for (int i = 1; i < N; ++i) {
            L += (pl.points[i] - pl.points[i-1]).norm();
        }
        if (addWrap) L += (pl.points.front() - pl.points.back()).norm();
        return L;
    };

    std::vector<Polyline> resampled;
    resampled.reserve(polys.size());
    stats.perPoly.reserve(polys.size());
    for (const auto& pl : polys) {
        Polyline rp;
        bool fastPath = resamplePolyline(pl, L_target, rp, disableFastPath);

        ResamplePolyStat ps;
        ps.chainId   = pl.chainId;
        ps.family    = pl.family;
        ps.isClosed  = rp.isClosed;
        ps.fastPath  = fastPath;
        // Input closed loops: buildOrderedPolylines appends the start
        // vertex at the end (walk only breaks *after* pushing startPt via
        // nextOther), so the wrap edge is already counted in the N
        // consecutive differences of the N+1 points. addWrap=false here.
        //
        // Resampled closed loops (including fast-path): stored as N unique
        // samples; the fast-path strips the input's duplicated trailing
        // start for exactly this reason. polylinesToSegments closes the
        // loop via (k+1)%N, so we must add the wrap edge manually to
        // match the arclength that actually reaches R.segments.
        ps.beforeLen = polylineArclen(pl, /*addWrap=*/false);
        ps.afterLen  = polylineArclen(rp, /*addWrap=*/rp.isClosed);
        stats.perPoly.push_back(ps);

        if (rp.isClosed) {
            ++stats.closedLoops;
            if (fastPath) ++stats.closedFastPath;
            else          ++stats.closedResampled;
        } else {
            ++stats.openChains;
            if (fastPath) ++stats.openFastPath;
            else          ++stats.openResampled;
        }
        resampled.push_back(std::move(rp));
    }
    stats.outputPolylines = (int)resampled.size();

    auto out = polylinesToSegments(resampled);
    stats.outputSegs = (int)out.size();

    detail::segmentLengthStats(in,  stats.inputSegLenMean,  stats.inputSegLenStd,
                               stats.inputSegLenMin,  stats.inputSegLenMax,
                               stats.inputTotalArclen);
    detail::segmentLengthStats(out, stats.outputSegLenMean, stats.outputSegLenStd,
                               stats.outputSegLenMin, stats.outputSegLenMax,
                               stats.outputTotalArclen);
    return out;
}

} // namespace wgf
