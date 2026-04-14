// isolines.h — extract θ isolines (zero-crossings of Re(θ)) per triangle.
//
// θ is stored as a unit complex per vertex. For each face we look for
// sign changes of Re(θ) along its three edges; a triangle crossed by
// the zero level set emits exactly one segment (two crossing points
// linearly interpolated along the edges).
//
// The output is a list of segments. Each segment carries the cover
// face index that produced it so that a later projection step can map
// it back to its base face and kagome family (layer mod 3).

#pragma once

#include "mesh.h"
#include "cover.h"

namespace wgf {

struct Segment {
    Eigen::Vector3d a, b;   // 3D endpoints (on the cover = on the base)
    int faceIdx;            // face in the mesh passed to extractIsolines
};

inline std::vector<Segment> extractIsolines(const Mesh& m, const Vec& theta) {
    std::vector<Segment> out;
    const int nF = m.nF();
    out.reserve(nF / 2);

    for (int f = 0; f < nF; ++f) {
        int v[3] = { m.F(f, 0), m.F(f, 1), m.F(f, 2) };
        double r[3] = { theta[2*v[0]], theta[2*v[1]], theta[2*v[2]] };
        Eigen::Vector3d p[3] = {
            m.V.row(v[0]), m.V.row(v[1]), m.V.row(v[2])
        };

        int nCross = 0;
        Eigen::Vector3d cp[2];
        for (int e = 0; e < 3; ++e) {
            int a = e, b = (e + 1) % 3;
            double ra = r[a], rb = r[b];
            if ((ra <= 0 && rb > 0) || (rb <= 0 && ra > 0)) {
                double t = ra / (ra - rb);
                if (nCross < 2) cp[nCross] = p[a] + t * (p[b] - p[a]);
                nCross++;
            }
        }
        if (nCross == 2) {
            Segment s;
            s.a = cp[0];
            s.b = cp[1];
            s.faceIdx = f;
            out.push_back(s);
        }
    }
    return out;
}

// Segment with base-family assignment.
struct ProjectedSegment {
    Eigen::Vector3d a, b;
    int baseFaceIdx;
    int family;       // 0, 1, or 2
};

inline std::vector<ProjectedSegment> projectToBase(
    const std::vector<Segment>& segs,
    const CoverBuildResult& cov) {

    std::vector<ProjectedSegment> out;
    out.reserve(segs.size());
    for (const auto& s : segs) {
        int cf = s.faceIdx;
        int bf = cov.coverFaceBaseF[cf];
        int lay = cov.coverFaceLayer[cf];
        ProjectedSegment p;
        p.a = s.a;
        p.b = s.b;
        p.baseFaceIdx = bf;
        p.family = lay % 3;
        out.push_back(p);
    }
    return out;
}

} // namespace wgf
