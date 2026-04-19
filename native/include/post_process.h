// post_process.h — isoline post-processing (Vekhter 2019 §5.2).
//
// This header implements the first three post-processing steps from §5.2:
//
//   1. "resample the isolines so that all segments are approximately the
//       same length"
//   2. "cut the ribbons into two pieces at points with very high amounts
//       of geodesic curvature"
//   3. "prune very short ribbon segments"
//
// Step 4 ("extend to crossings") is intentionally NOT implemented here;
// it will follow in its own PR (Phase E-3).
//
// Pipeline position: runs AFTER the centerline extraction emits its
// ProjectedSegments (one per traced isoline face-crossing) and BEFORE
// the pipeline returns. Takes raw ProjectedSegments in, returns
// resampled-cut-pruned ProjectedSegments.
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
//   D. For each polyline, compute a discrete geodesic curvature proxy
//      (turning-angle / local-arclength) at interior points, decide a
//      per-chain threshold (adaptive 95th percentile clamped by absolute
//      caps at π/(4·L_target) and π/L_target), and split the polyline
//      at every point whose kappa exceeds the threshold. Sub-chains
//      share the cut point. Closed loops with >=1 cut become open chains.
//   E. Drop any surviving polyline whose total arclength is below a
//      per-chain minimum (default 3 · L_target). This prunes the short
//      "carpet scraps" that §5.2 calls out.

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

// ---------------------------------------------------------------------------
// Step D / E: curvature cut + short prune (Vekhter §5.2, steps 2 & 3).
// ---------------------------------------------------------------------------

struct CutPruneOptions {
    // Curvature cut (§5.2 step 2)
    //
    // cutThresholdAbs > 0 : use this absolute kappa threshold for every
    //                       chain (bypasses the per-chain adaptive rule).
    // cutThresholdAbs <= 0: per-chain adaptive percentile, clamped to the
    //                       interval [kappaMinCap, kappaMaxCap] where
    //                       kappaMinCap = π / (4·L_target),
    //                       kappaMaxCap = π /     L_target.
    double cutThresholdAbs = 0.0;
    double cutPercentile   = 0.95;
    bool   disableCut      = false;

    // Short prune (§5.2 step 3)
    //
    // minChainLengthAbs > 0 : use this absolute minimum arclength.
    // minChainLengthAbs <= 0: use minChainMult · L_target.
    double minChainLengthAbs = 0.0;
    double minChainMult      = 3.0;
    bool   disablePrune      = false;

    double L_target = 1.0;
};

struct CutPruneStats {
    // Counts across the full pass.
    int inputPolylines      = 0;
    int afterCutPolylines   = 0;
    int afterPrunePolylines = 0;
    int cutsApplied         = 0;
    int prunedPolylines     = 0;

    // Kappa distribution over all interior points of all input polylines.
    int    totalKappaSamples = 0;
    double kappaP50 = 0.0, kappaP75 = 0.0, kappaP90 = 0.0;
    double kappaP95 = 0.0, kappaP99 = 0.0, kappaMax = 0.0;

    // Per-chain threshold distribution (adaptive path only).
    int    thresholdSamples = 0;
    double thrMin  = 0.0, thrMax = 0.0, thrMean = 0.0;

    // Length histograms (bucketed by multiples of L_target).
    // Buckets: [<1·L, <2·L, <5·L, <10·L, >=10·L]
    int lenHistBefore[5] = {0, 0, 0, 0, 0};
    int lenHistAfter [5] = {0, 0, 0, 0, 0};

    // Arclength totals (for the summary report).
    double inputArclen = 0.0, cutArclen = 0.0, outputArclen = 0.0;

    // Family counts after prune (for balance check).
    int familyCountsAfter[3] = {0, 0, 0};

    // Kappa distribution over the OUTPUT interior points (sanity check:
    // should have p99 below the threshold after a successful cut).
    double outKappaP95 = 0.0, outKappaP99 = 0.0, outKappaMax = 0.0;

    // Echo of the effective configuration used.
    double effectiveMinChainLength = 0.0;
    bool   cutWasAdaptive          = true;
};

namespace detail {

// Discrete geodesic curvature proxy at each "interior" point of a polyline.
//
// Open chain: kappa defined for i = 1 .. N-2 → returns N-2 values
//             (element i corresponds to points[i+1]).
// Closed loop: kappa defined for every i = 0 .. N-1 using the cyclic
//             previous / next indices → returns N values (element i
//             corresponds to points[i]).
//
// "Interior" index convention (what the caller uses to cut):
//   open  : cut indices live in [1, N-2], kappa[k] ↔ cut index k+1
//   closed: cut indices live in [0, N-1], kappa[k] ↔ cut index k
inline std::vector<double> geodesicCurvatureProxy(const Polyline& pl) {
    std::vector<double> kappa;
    const int N = (int)pl.points.size();
    if (N < 3) return kappa;

    auto kappaAt = [&](int prev, int cur, int next) -> double {
        Eigen::Vector3d v1 = pl.points[cur]  - pl.points[prev];
        Eigen::Vector3d v2 = pl.points[next] - pl.points[cur];
        double L1 = v1.norm();
        double L2 = v2.norm();
        if (L1 < 1e-12 || L2 < 1e-12) return 0.0;
        v1 /= L1;
        v2 /= L2;
        double dot_val = std::clamp(v1.dot(v2), -1.0, 1.0);
        double angle = std::acos(dot_val);
        double L_local = 0.5 * (L1 + L2);
        return angle / std::max(L_local, 1e-9);
    };

    if (pl.isClosed) {
        kappa.reserve(N);
        for (int i = 0; i < N; ++i) {
            int ip = (i - 1 + N) % N;
            int in = (i + 1) % N;
            kappa.push_back(kappaAt(ip, i, in));
        }
    } else {
        kappa.reserve(N - 2);
        for (int i = 1; i < N - 1; ++i) {
            kappa.push_back(kappaAt(i - 1, i, i + 1));
        }
    }
    return kappa;
}

inline double polylineArclength(const Polyline& pl) {
    const int N = (int)pl.points.size();
    if (N < 2) return 0.0;
    double L = 0.0;
    for (int i = 1; i < N; ++i) L += (pl.points[i] - pl.points[i-1]).norm();
    if (pl.isClosed) L += (pl.points.front() - pl.points.back()).norm();
    return L;
}

// Decide a per-chain threshold given the chain's kappa values.
// Returns +inf when the chain has no interior points (no candidate cuts).
inline double decideCutThreshold(
    const std::vector<double>& kappas,
    double percentile,
    double L_target)
{
    if (kappas.empty()) return std::numeric_limits<double>::infinity();
    std::vector<double> sorted = kappas;
    std::sort(sorted.begin(), sorted.end());
    // Floor-index: for a 20-element vector and p=0.95, we want index 19
    // (the largest). std::min guards against percentile=1.0.
    int idx = (int)std::floor(percentile * (double)sorted.size());
    if (idx < 0) idx = 0;
    if (idx >= (int)sorted.size()) idx = (int)sorted.size() - 1;
    double p = sorted[idx];
    double kmin = M_PI / (4.0 * std::max(L_target, 1e-9));
    double kmax = M_PI /        std::max(L_target, 1e-9);
    return std::clamp(p, kmin, kmax);
}

// Cut an open polyline at a sorted set of interior indices (each in [1, N-2]).
// Produces K+1 open sub-polylines that share each cut point between
// neighbouring sub-chains. Face ids are partitioned along the cuts.
inline std::vector<Polyline> cutOpenAtIndices(
    const Polyline& pl, const std::vector<int>& cuts)
{
    const int N = (int)pl.points.size();
    const int M = (int)pl.faceIds.size();
    std::vector<Polyline> out;
    if (cuts.empty()) { out.push_back(pl); return out; }

    // Sentinel 0 on the left and N-1 on the right; each sub-chain spans
    // bounds[j]..bounds[j+1] inclusive.
    std::vector<int> bounds;
    bounds.reserve(cuts.size() + 2);
    bounds.push_back(0);
    for (int c : cuts) bounds.push_back(c);
    bounds.push_back(N - 1);

    for (std::size_t j = 0; j + 1 < bounds.size(); ++j) {
        int a = bounds[j];
        int b = bounds[j + 1];
        if (b <= a) continue;
        Polyline sub;
        sub.chainId  = pl.chainId;
        sub.family   = pl.family;
        sub.isClosed = false;
        sub.points.assign(pl.points.begin() + a, pl.points.begin() + b + 1);
        int faceLo = a;
        int faceHi = std::min(b, M);           // faceIds are [0, M)
        if (faceHi > faceLo) {
            sub.faceIds.assign(pl.faceIds.begin() + faceLo,
                               pl.faceIds.begin() + faceHi);
        }
        out.push_back(std::move(sub));
    }
    return out;
}

// Cut a closed polyline at a sorted set of indices (each in [0, N-1]).
// With K >= 1 cuts, the loop becomes K open sub-polylines (cyclic pieces).
// With K == 0 the loop is returned unchanged.
//
// For K == 1, we emit one open sub-polyline that wraps once around the
// loop back to the (shared) cut point.
inline std::vector<Polyline> cutClosedAtIndices(
    const Polyline& pl, const std::vector<int>& cuts)
{
    const int N = (int)pl.points.size();
    std::vector<Polyline> out;
    if (cuts.empty()) { out.push_back(pl); return out; }

    const int K = (int)cuts.size();
    // faceIds.size() should be == N on resampled closed loops (one per
    // segment, including the wrap-around edge). Fall back gracefully if
    // the invariant is ever violated.
    const int M = (int)pl.faceIds.size();

    // Number of forward edges covered per sub-chain. For K >= 2 the cuts
    // carve the cycle into K arcs; for K == 1 the whole cycle becomes one
    // open arc of length N that starts and ends at the single cut point.
    for (int j = 0; j < K; ++j) {
        int start = cuts[j];
        int end   = cuts[(j + 1) % K];
        int edgeCount;
        if (K == 1) {
            edgeCount = N;                      // full wrap
        } else {
            edgeCount = (end - start + N) % N;  // forward distance mod N
            if (edgeCount == 0) edgeCount = N;  // colocated cuts => full wrap
        }
        Polyline sub;
        sub.chainId  = pl.chainId;
        sub.family   = pl.family;
        sub.isClosed = false;
        int idx = start;
        sub.points.push_back(pl.points[idx]);
        for (int step = 0; step < edgeCount; ++step) {
            if (M > 0) sub.faceIds.push_back(pl.faceIds[idx % M]);
            idx = (idx + 1) % N;
            sub.points.push_back(pl.points[idx]);
        }
        out.push_back(std::move(sub));
    }
    return out;
}

// Normalize a closed-loop polyline to the compact N-unique-points
// convention (no duplicated start at the end). buildOrderedPolylines
// emits closed loops in "convention A" — N+1 points where points.back()
// quantizes to points.front() — which would confuse the cyclic index
// math in geodesicCurvatureProxy / cutClosedAtIndices. After this call,
// points.size() == N, faceIds.size() == N (one per edge, including the
// implicit wrap-around edge from points[N-1] to points[0]). Open chains
// are left untouched.
inline void normalizeClosedLoopCompact(Polyline& pl) {
    if (!pl.isClosed) return;
    const int N = (int)pl.points.size();
    if (N < 2) return;
    if (quantKey(pl.points.front()) == quantKey(pl.points.back())) {
        pl.points.pop_back();
    }
}

inline int lengthBucketIndex(double L, double L_target) {
    double r = L / std::max(L_target, 1e-30);
    if (r < 1.0)  return 0;
    if (r < 2.0)  return 1;
    if (r < 5.0)  return 2;
    if (r < 10.0) return 3;
    return 4;
}

} // namespace detail

// Apply cut + prune to a batch of Polylines. Returns the surviving
// polylines (post-prune), with chainIds rewritten so that cut sub-chains
// get fresh ids from (maxInputChainId + 1) onwards and un-cut chains
// keep their original id.
inline std::vector<Polyline> cutAndPrunePolylines(
    std::vector<Polyline> polys,
    const CutPruneOptions& opt,
    CutPruneStats&         stats)
{
    stats = CutPruneStats{};
    stats.inputPolylines = (int)polys.size();
    stats.effectiveMinChainLength =
        (opt.minChainLengthAbs > 0.0)
            ? opt.minChainLengthAbs
            : opt.minChainMult * opt.L_target;
    stats.cutWasAdaptive = (opt.cutThresholdAbs <= 0.0);

    // Normalize closed loops to the compact convention so that cyclic
    // index math in the curvature proxy and the cut helpers lines up.
    for (auto& pl : polys) detail::normalizeClosedLoopCompact(pl);

    // --- Pass 1: gather kappa statistics over the whole batch -----------
    std::vector<double> allKappas;
    for (const auto& pl : polys) {
        for (double k : detail::geodesicCurvatureProxy(pl)) allKappas.push_back(k);
    }
    stats.totalKappaSamples = (int)allKappas.size();
    if (!allKappas.empty()) {
        std::vector<double> sorted = allKappas;
        std::sort(sorted.begin(), sorted.end());
        auto pick = [&](double p) {
            int idx = (int)std::floor(p * (double)sorted.size());
            if (idx < 0) idx = 0;
            if (idx >= (int)sorted.size()) idx = (int)sorted.size() - 1;
            return sorted[idx];
        };
        stats.kappaP50 = pick(0.50);
        stats.kappaP75 = pick(0.75);
        stats.kappaP90 = pick(0.90);
        stats.kappaP95 = pick(0.95);
        stats.kappaP99 = pick(0.99);
        stats.kappaMax = sorted.back();
    }

    // --- Pass 2: cut -----------------------------------------------------
    int maxChainId = -1;
    for (const auto& pl : polys) if (pl.chainId > maxChainId) maxChainId = pl.chainId;
    int nextChainId = maxChainId + 1;

    std::vector<Polyline> cutOut;
    cutOut.reserve(polys.size());
    double sumThr = 0.0;
    double thrMinAcc =  std::numeric_limits<double>::infinity();
    double thrMaxAcc = -std::numeric_limits<double>::infinity();
    int    thrCount = 0;

    stats.inputArclen = 0.0;
    for (const auto& pl : polys) {
        stats.inputArclen += detail::polylineArclength(pl);
    }

    for (const auto& pl : polys) {
        const int N = (int)pl.points.size();
        if (opt.disableCut || N < 3) {
            cutOut.push_back(pl);
            continue;
        }

        auto kappas = detail::geodesicCurvatureProxy(pl);
        if (kappas.empty()) {
            cutOut.push_back(pl);
            continue;
        }

        double thr;
        if (opt.cutThresholdAbs > 0.0) {
            thr = opt.cutThresholdAbs;
        } else {
            thr = detail::decideCutThreshold(kappas, opt.cutPercentile, opt.L_target);
            if (std::isfinite(thr)) {
                sumThr += thr;
                thrMinAcc = std::min(thrMinAcc, thr);
                thrMaxAcc = std::max(thrMaxAcc, thr);
                ++thrCount;
            }
        }

        // Translate kappa-array indices to polyline point indices.
        std::vector<int> cutIndices;
        cutIndices.reserve(kappas.size());
        if (pl.isClosed) {
            for (int k = 0; k < (int)kappas.size(); ++k) {
                if (kappas[k] > thr) cutIndices.push_back(k);
            }
        } else {
            // kappa[k] corresponds to point index k+1 in [1, N-2].
            for (int k = 0; k < (int)kappas.size(); ++k) {
                if (kappas[k] > thr) cutIndices.push_back(k + 1);
            }
        }

        if (cutIndices.empty()) {
            cutOut.push_back(pl);
            continue;
        }

        std::vector<Polyline> subs = pl.isClosed
            ? detail::cutClosedAtIndices(pl, cutIndices)
            : detail::cutOpenAtIndices(pl, cutIndices);

        // Assign fresh chainIds to every sub-chain produced by a cut.
        // Skip degenerate (<2 pts) remnants so they don't inflate counts.
        for (auto& sub : subs) {
            if ((int)sub.points.size() < 2) continue;
            sub.chainId = nextChainId++;
            cutOut.push_back(std::move(sub));
        }
        stats.cutsApplied += (int)cutIndices.size();
    }
    stats.afterCutPolylines = (int)cutOut.size();
    stats.cutArclen = 0.0;
    for (const auto& pl : cutOut) stats.cutArclen += detail::polylineArclength(pl);
    if (thrCount > 0) {
        stats.thresholdSamples = thrCount;
        stats.thrMin  = thrMinAcc;
        stats.thrMax  = thrMaxAcc;
        stats.thrMean = sumThr / (double)thrCount;
    }

    // Length histogram (before prune).
    for (const auto& pl : cutOut) {
        double L = detail::polylineArclength(pl);
        int b = detail::lengthBucketIndex(L, opt.L_target);
        if (b >= 0 && b < 5) ++stats.lenHistBefore[b];
    }

    // --- Pass 3: prune ---------------------------------------------------
    std::vector<Polyline> finalOut;
    finalOut.reserve(cutOut.size());
    const double Lmin = stats.effectiveMinChainLength;
    for (auto& pl : cutOut) {
        double L = detail::polylineArclength(pl);
        if (!opt.disablePrune && L < Lmin) {
            ++stats.prunedPolylines;
            continue;
        }
        finalOut.push_back(std::move(pl));
    }
    stats.afterPrunePolylines = (int)finalOut.size();

    for (const auto& pl : finalOut) {
        double L = detail::polylineArclength(pl);
        int b = detail::lengthBucketIndex(L, opt.L_target);
        if (b >= 0 && b < 5) ++stats.lenHistAfter[b];
        stats.outputArclen += L;
        if (pl.family >= 0 && pl.family < 3) ++stats.familyCountsAfter[pl.family];
    }

    // Output-side kappa sanity stats.
    {
        std::vector<double> outK;
        for (const auto& pl : finalOut) {
            for (double k : detail::geodesicCurvatureProxy(pl)) outK.push_back(k);
        }
        if (!outK.empty()) {
            std::sort(outK.begin(), outK.end());
            auto pick = [&](double p) {
                int idx = (int)std::floor(p * (double)outK.size());
                if (idx < 0) idx = 0;
                if (idx >= (int)outK.size()) idx = (int)outK.size() - 1;
                return outK[idx];
            };
            stats.outKappaP95 = pick(0.95);
            stats.outKappaP99 = pick(0.99);
            stats.outKappaMax = outK.back();
        }
    }

    return finalOut;
}

// Convenience wrapper: rebuild polylines from ProjectedSegments, run
// cut + prune, re-emit ProjectedSegments with the rewritten chainIds.
inline std::vector<ProjectedSegment> cutAndPruneProjectedSegments(
    const std::vector<ProjectedSegment>& in,
    const CutPruneOptions&               opt,
    CutPruneStats&                       stats)
{
    auto polys = buildOrderedPolylines(in);
    auto kept  = cutAndPrunePolylines(std::move(polys), opt, stats);
    return polylinesToSegments(kept);
}

} // namespace wgf
