// face_topology.h — face-level geometry helpers for Phase E-3.
//
// Helpers needed by the upcoming "extend to crossings" post-processing
// step (Vekhter 2019 §5.2, step 4). None of this is wired into the
// pipeline yet; follow-up PRs (E-3a-2..E-3a-4) consume these functions.
//
// Functions:
//   - oppositeFaceAcross:   step across a half-edge to the neighboring face
//   - computeExitPoint:     ray-march a point on a face along a tangent,
//                           reporting which edge it exits through
//   - segSegIntersectOnFace: 3D segment–segment intersection restricted to
//                            a shared face plane
//
// All functions operate on the half-edge `Mesh` from mesh.h and use Eigen.

#pragma once

#include "mesh.h"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <optional>

namespace wgf {

// ---------------------------------------------------------------------------
// ExitInfo: result of ray-marching a point across a face.
// ---------------------------------------------------------------------------
struct ExitInfo {
    Eigen::Vector3d point;   // 3D coordinate where the ray exits the face
    int             exitEdge;  // half-edge index of the edge we exited through
    double          distance;  // euclidean distance from the input pos to point
};

// ---------------------------------------------------------------------------
// oppositeFaceAcross
// ---------------------------------------------------------------------------
// Given a half-edge `edgeIdx` incident to `currentFace` (directly or via its
// twin), return the face on the opposite side. Returns -1 if the edge is on
// the mesh boundary (no twin).
//
// Either half-edge of the shared edge may be passed in — we pick whichever
// side is not `currentFace`. This lets callers forward raw half-edges from
// exit-point queries without having to check orientation first.
inline int oppositeFaceAcross(const Mesh& mesh,
                              int edgeIdx,
                              int currentFace) {
    const int twin = mesh.he[edgeIdx].twin;
    if (twin < 0) return -1;
    const int fA = mesh.he[edgeIdx].face;
    const int fB = mesh.he[twin].face;
    return (fA == currentFace) ? fB : fA;
}

// ---------------------------------------------------------------------------
// computeExitPoint
// ---------------------------------------------------------------------------
// Ray-march from `pos` (assumed to lie in the plane of `face`) along
// `tangent` (3D vector, not required to be unit length or lie in the face
// plane — we project it onto the face's tangent plane first) and report
// which of the face's three edges the ray exits through.
//
// Returns nullopt if:
//   - the projected tangent is (numerically) zero (tangent ∥ normal), or
//   - no forward intersection is found (pos is outside the triangle and the
//     ray points away from it).
inline std::optional<ExitInfo> computeExitPoint(const Mesh& mesh,
                                                int face,
                                                const Eigen::Vector3d& pos,
                                                const Eigen::Vector3d& tangent) {
    // --- 1. Face geometry: three vertices and the unit normal. -----------
    const Eigen::Vector3d v0 = mesh.V.row(mesh.F(face, 0));
    const Eigen::Vector3d v1 = mesh.V.row(mesh.F(face, 1));
    const Eigen::Vector3d v2 = mesh.V.row(mesh.F(face, 2));
    Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
    const double nlen = n.norm();
    if (nlen < 1e-30) return std::nullopt;
    n /= nlen;

    // --- 2. Project tangent onto the face plane. --------------------------
    Eigen::Vector3d dir = tangent - tangent.dot(n) * n;
    const double dlen = dir.norm();
    if (dlen < 1e-12) return std::nullopt;
    dir /= dlen;

    // --- 3. 2D basis on the face plane. -----------------------------------
    Eigen::Vector3d e1 = (v1 - v0).normalized();
    Eigen::Vector3d e2 = n.cross(e1).normalized();

    auto to2D = [&](const Eigen::Vector3d& p) {
        Eigen::Vector3d d = p - v0;
        return Eigen::Vector2d(d.dot(e1), d.dot(e2));
    };

    const Eigen::Vector2d p2   = to2D(pos);
    const Eigen::Vector2d d2(dir.dot(e1), dir.dot(e2));
    const Eigen::Vector2d V2[3] = { to2D(v0), to2D(v1), to2D(v2) };

    // --- 4. Walk the three half-edges of this face and intersect. --------
    // Face half-edges in CCW order, starting at f2he[face].
    const int h0 = mesh.f2he[face];
    const int h1 = mesh.he[h0].next;
    const int h2 = mesh.he[h1].next;
    const int hes[3] = { h0, h1, h2 };

    // Each half-edge `h` goes from heStart(h) (== one triangle vertex) to
    // he[h].vertex (== the next triangle vertex). The F-matrix order is
    // F(face,0..2). We need to map each half-edge to the corresponding
    // (a2, b2) endpoint pair in V2[] so callers can look up which edge was
    // hit later via the returned half-edge index.
    auto vert2D = [&](int vIdx) -> Eigen::Vector2d {
        if (vIdx == mesh.F(face, 0)) return V2[0];
        if (vIdx == mesh.F(face, 1)) return V2[1];
        return V2[2];
    };

    constexpr double kParallelEps = 1e-12;
    constexpr double kForwardEps  = 1e-9;
    constexpr double kEdgeEps     = 1e-9;

    double bestT = std::numeric_limits<double>::infinity();
    int    bestH = -1;
    Eigen::Vector2d bestHit2D;

    for (int k = 0; k < 3; ++k) {
        const int h = hes[k];
        const Eigen::Vector2d a2 = vert2D(mesh.heStart(h));
        const Eigen::Vector2d b2 = vert2D(mesh.he[h].vertex);
        const Eigen::Vector2d ed = b2 - a2;

        // Solve p2 + t*d2 = a2 + s*ed for (t, s) using a 2x2 system.
        // Matrix: [ d2.x , -ed.x ] [t]   [ a2.x - p2.x ]
        //         [ d2.y , -ed.y ] [s] = [ a2.y - p2.y ]
        const double denom = d2.x() * (-ed.y()) - d2.y() * (-ed.x());
        if (std::abs(denom) < kParallelEps) continue;  // ray ∥ edge

        const double rx = a2.x() - p2.x();
        const double ry = a2.y() - p2.y();
        const double t = (rx * (-ed.y()) - ry * (-ed.x())) / denom;
        const double s = (d2.x() * ry - d2.y() * rx) / denom;

        if (t < kForwardEps) continue;                        // not forward
        if (s < -kEdgeEps || s > 1.0 + kEdgeEps) continue;    // off-edge
        if (t < bestT) {
            bestT     = t;
            bestH     = h;
            bestHit2D = a2 + s * ed;
        }
    }

    if (bestH < 0) return std::nullopt;

    // --- 5. Lift the 2D hit back to 3D. ----------------------------------
    Eigen::Vector3d hit3D = v0 + bestHit2D.x() * e1 + bestHit2D.y() * e2;
    ExitInfo info;
    info.point    = hit3D;
    info.exitEdge = bestH;
    info.distance = (hit3D - pos).norm();
    return info;
}

// ---------------------------------------------------------------------------
// segSegIntersectOnFace
// ---------------------------------------------------------------------------
// Intersect segments a0→a1 and b0→b1, both assumed to lie on the face whose
// unit normal is `faceNormal`. Returns the 3D intersection point if the
// segments cross (including T-junction endpoint contacts, within a small
// numerical tolerance). Returns nullopt if the segments are parallel or do
// not intersect within their extents.
inline std::optional<Eigen::Vector3d> segSegIntersectOnFace(
    const Eigen::Vector3d& a0, const Eigen::Vector3d& a1,
    const Eigen::Vector3d& b0, const Eigen::Vector3d& b1,
    const Eigen::Vector3d& faceNormal) {
    const Eigen::Vector3d da  = a1 - a0;
    const Eigen::Vector3d db  = b1 - b0;
    const Eigen::Vector3d dab = b0 - a0;

    // denom = (da × db) · n. Sign-aware so we can invert below.
    const double denom = da.cross(db).dot(faceNormal);
    constexpr double kParallelEps = 1e-12;
    if (std::abs(denom) < kParallelEps) return std::nullopt;

    const double t = dab.cross(db).dot(faceNormal) / denom;
    const double s = dab.cross(da).dot(faceNormal) / denom;

    constexpr double kEndpointEps = 1e-9;
    if (t < -kEndpointEps || t > 1.0 + kEndpointEps) return std::nullopt;
    if (s < -kEndpointEps || s > 1.0 + kEndpointEps) return std::nullopt;

    return a0 + t * da;
}

} // namespace wgf
