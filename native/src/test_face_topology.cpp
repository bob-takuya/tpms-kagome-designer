// Unit tests for face_topology.h (Phase E-3 helpers).
//
// Standalone executable — no external test framework. Uses a small
// hand-built mesh (two triangles sharing an edge in the z=0 plane) to
// exercise each helper. Each check prints PASS/FAIL and the program exits
// non-zero if any assertion fails.

#include "face_topology.h"
#include "mesh.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace wgf;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, msg); \
        } else {                                                               \
            std::fprintf(stdout, "PASS  %s\n", msg);                           \
        }                                                                      \
    } while (0)

// Build the two-triangle square mesh used by every test:
//
//     v3 ---- v2
//     |  \ f1  |
//     |    \   |
//     | f0   \ |
//     v0 ---- v1
//
// Vertices: v0=(0,0,0), v1=(1,0,0), v2=(1,1,0), v3=(0,1,0)
// Face f0 = (v0,v1,v2)  (CCW when viewed from +z)
// Face f1 = (v0,v2,v3)  (CCW when viewed from +z)
// Shared (interior) edge: v0–v2.
Mesh buildSquareMesh() {
    Mesh m;
    m.V.resize(4, 3);
    m.V.row(0) << 0.0, 0.0, 0.0;
    m.V.row(1) << 1.0, 0.0, 0.0;
    m.V.row(2) << 1.0, 1.0, 0.0;
    m.V.row(3) << 0.0, 1.0, 0.0;

    m.F.resize(2, 3);
    m.F.row(0) << 0, 1, 2;
    m.F.row(1) << 0, 2, 3;

    buildHalfEdges(m);
    return m;
}

// Find the half-edge in `face` that connects `va` → `vb`.
int findHalfEdge(const Mesh& m, int face, int va, int vb) {
    const int h0 = m.f2he[face];
    int h = h0;
    for (int i = 0; i < 3; ++i) {
        if (m.heStart(h) == va && m.he[h].vertex == vb) return h;
        h = m.he[h].next;
    }
    return -1;
}

// -------------------------------------------------------------------------
// Test 1: oppositeFaceAcross
// -------------------------------------------------------------------------
void testOppositeFaceAcross() {
    std::fprintf(stdout, "\n=== Test 1: oppositeFaceAcross ===\n");
    Mesh m = buildSquareMesh();

    // Shared interior edge between f0=(0,1,2) and f1=(0,2,3): in f0's CCW
    // orientation the half-edge goes v2 → v0; its twin v0 → v2 lies in f1.
    const int hInterior = findHalfEdge(m, 0, 2, 0);
    CHECK(hInterior >= 0, "interior half-edge v2->v0 found in f0");
    CHECK(m.he[hInterior].twin >= 0, "interior half-edge has a twin");

    const int opp = oppositeFaceAcross(m, hInterior, /*currentFace=*/0);
    CHECK(opp == 1, "interior edge: opposite of f0 across shared edge is f1");

    // Passing the same half-edge but with the twin's face as "current" should
    // return f0 (we use either side symmetrically).
    const int oppSym = oppositeFaceAcross(m, hInterior, /*currentFace=*/1);
    CHECK(oppSym == 0, "interior edge: symmetric query returns f0");

    // Boundary edge: any of v0→v1, v1→v2, v2→v3, v3→v0. Pick v0→v1 in f0.
    const int hBoundary = findHalfEdge(m, 0, 0, 1);
    CHECK(hBoundary >= 0, "boundary half-edge v0->v1 found in f0");
    CHECK(m.he[hBoundary].twin == -1, "boundary half-edge has no twin");

    const int oppBoundary = oppositeFaceAcross(m, hBoundary, /*currentFace=*/0);
    CHECK(oppBoundary == -1, "boundary edge: opposite face is -1");
}

// -------------------------------------------------------------------------
// Test 2: computeExitPoint
// -------------------------------------------------------------------------
void testComputeExitPoint() {
    std::fprintf(stdout, "\n=== Test 2: computeExitPoint ===\n");
    Mesh m = buildSquareMesh();

    // Centroid of face 0 = (v0+v1+v2)/3 = (2/3, 1/3, 0).
    const Eigen::Vector3d c0(2.0 / 3.0, 1.0 / 3.0, 0.0);

    // 2a. Shoot +x from the centroid. The triangle v0-v1-v2 has a hypotenuse
    //     x + (−1) y = 1 (i.e., edge v1-v2 is x=1-?... wait: v1=(1,0), v2=(1,1)).
    //     Actually v1-v2 lies on x=1; shooting +x from (2/3, 1/3) hits x=1 at
    //     (1, 1/3), crossing the right boundary v1-v2.
    {
        auto exit = computeExitPoint(m, /*face=*/0, c0, Eigen::Vector3d(1, 0, 0));
        CHECK(exit.has_value(), "centroid +x: exit found");
        if (exit) {
            CHECK(std::abs(exit->point.x() - 1.0) < 1e-9,
                  "centroid +x: exits at x=1");
            CHECK(std::abs(exit->point.y() - 1.0 / 3.0) < 1e-9,
                  "centroid +x: exits at y=1/3");
            CHECK(std::abs(exit->distance - 1.0 / 3.0) < 1e-9,
                  "centroid +x: distance is 1/3");
            // The exit edge should be v1→v2 (half-edge inside face 0).
            const int hv1v2 = findHalfEdge(m, 0, 1, 2);
            CHECK(exit->exitEdge == hv1v2,
                  "centroid +x: exit edge is v1->v2");
        }
    }

    // 2b. Shoot toward v2 direction (+x,+y normalized) from centroid. v2 itself
    //     is a triangle vertex — the ray should exit at v2 (or extremely close)
    //     through one of the two edges incident to v2. We just sanity-check
    //     that an exit exists and the hit is at v2.
    {
        Eigen::Vector3d dir = (Eigen::Vector3d(1, 1, 0) - c0).normalized();
        auto exit = computeExitPoint(m, 0, c0, dir);
        CHECK(exit.has_value(), "centroid toward v2: exit found");
        if (exit) {
            CHECK((exit->point - Eigen::Vector3d(1, 1, 0)).norm() < 1e-9,
                  "centroid toward v2: hit is at v2");
        }
    }

    // 2c. Degenerate: tangent ∥ face normal → nullopt.
    {
        auto exit = computeExitPoint(m, 0, c0, Eigen::Vector3d(0, 0, 1));
        CHECK(!exit.has_value(), "tangent ∥ normal: returns nullopt");
    }

    // 2d. Zero tangent → nullopt.
    {
        auto exit = computeExitPoint(m, 0, c0, Eigen::Vector3d::Zero());
        CHECK(!exit.has_value(), "zero tangent: returns nullopt");
    }

    // 2e. Shoot −x from centroid → hits the hypotenuse edge v2→v0 (in face 0,
    //     the half-edge order is (0,1,2) so the third edge is v2->v0).
    {
        auto exit = computeExitPoint(m, 0, c0, Eigen::Vector3d(-1, 0, 0));
        CHECK(exit.has_value(), "centroid -x: exit found");
        if (exit) {
            // Line v0→v2 is y = x. Shooting −x from (2/3, 1/3) hits y=x at
            // (1/3, 1/3).
            CHECK(std::abs(exit->point.x() - 1.0 / 3.0) < 1e-9 &&
                  std::abs(exit->point.y() - 1.0 / 3.0) < 1e-9,
                  "centroid -x: exits at (1/3, 1/3)");
            const int hv2v0 = findHalfEdge(m, 0, 2, 0);
            CHECK(exit->exitEdge == hv2v0,
                  "centroid -x: exit edge is v2->v0 (shared)");
        }
    }
}

// -------------------------------------------------------------------------
// Test 3: segSegIntersectOnFace
// -------------------------------------------------------------------------
void testSegSegIntersectOnFace() {
    std::fprintf(stdout, "\n=== Test 3: segSegIntersectOnFace ===\n");
    const Eigen::Vector3d n(0, 0, 1);

    // 3a. Two diagonals of the unit square → cross at (0.5, 0.5, 0).
    {
        auto hit = segSegIntersectOnFace(
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 0),
            Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(1, 0, 0),
            n);
        CHECK(hit.has_value(), "diagonals: intersection found");
        if (hit) {
            CHECK((*hit - Eigen::Vector3d(0.5, 0.5, 0)).norm() < 1e-9,
                  "diagonals: intersect at (0.5, 0.5, 0)");
        }
    }

    // 3b. Parallel segments → nullopt.
    {
        auto hit = segSegIntersectOnFace(
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0),
            Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(1, 1, 0),
            n);
        CHECK(!hit.has_value(), "parallel segments: nullopt");
    }

    // 3c. Collinear (parallel with zero cross) → nullopt (degenerate denom).
    {
        auto hit = segSegIntersectOnFace(
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0),
            Eigen::Vector3d(0.5, 0, 0), Eigen::Vector3d(2, 0, 0),
            n);
        CHECK(!hit.has_value(), "collinear segments: nullopt");
    }

    // 3d. T-junction: endpoint of B touches interior of A at (0.5, 0, 0).
    {
        auto hit = segSegIntersectOnFace(
            Eigen::Vector3d(0, 0, 0),   Eigen::Vector3d(1, 0, 0),
            Eigen::Vector3d(0.5, 0, 0), Eigen::Vector3d(0.5, 1, 0),
            n);
        CHECK(hit.has_value(), "T-junction: intersection found");
        if (hit) {
            CHECK((*hit - Eigen::Vector3d(0.5, 0, 0)).norm() < 1e-9,
                  "T-junction: intersect at (0.5, 0, 0)");
        }
    }

    // 3e. Segments cross the extended lines but miss the actual segment extents.
    {
        auto hit = segSegIntersectOnFace(
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0),
            Eigen::Vector3d(2, -1, 0), Eigen::Vector3d(2, 1, 0),
            n);
        CHECK(!hit.has_value(), "disjoint segments whose lines would meet: nullopt");
    }

    // 3f. Works on a non-axis-aligned face. Tilt 45° about x: normal = (0,1,1)/√2.
    {
        const Eigen::Vector3d nn = Eigen::Vector3d(0, 1, 1).normalized();
        // Build two crossing segments lying in the plane z = 1 − y.
        // Use u = (1,0,0) and v = (0, 1/√2, -1/√2) as in-plane basis.
        const Eigen::Vector3d u(1, 0, 0);
        const Eigen::Vector3d v = nn.cross(u).normalized();
        const Eigen::Vector3d p0 = Eigen::Vector3d(0, 0.5, 0.5);
        auto onPlane = [&](double a, double b) {
            return p0 + a * u + b * v;
        };
        auto hit = segSegIntersectOnFace(
            onPlane(-1, -1), onPlane(1, 1),
            onPlane(-1,  1), onPlane(1, -1),
            nn);
        CHECK(hit.has_value(), "tilted plane: intersection found");
        if (hit) {
            CHECK((*hit - p0).norm() < 1e-9, "tilted plane: hit is at p0");
        }
    }
}

} // namespace

int main() {
    testOppositeFaceAcross();
    testComputeExitPoint();
    testSegSegIntersectOnFace();

    std::fprintf(stdout, "\n----\n");
    std::fprintf(stdout, "%d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
