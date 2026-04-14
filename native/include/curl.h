// curl.h — discrete curl operator C (paper §3.1, eq. 4).
//
// For each interior edge with shared 3D vector e_ij between faces fi, fj,
//
//     (C u)_e = u_fi · e_ij − u_fj · e_ij
//
// where each face's local 2D vector is interpreted inside that face's
// orthonormal frame. Each row of C has exactly four non-zeros:
//
//     row[2*fi + 0] = e_ij · e1_fi
//     row[2*fi + 1] = e_ij · e2_fi
//     row[2*fj + 0] = −(e_ij · e1_fj)
//     row[2*fj + 1] = −(e_ij · e2_fj)

#pragma once

#include "mesh.h"

namespace wgf {

inline SpMat buildCurlOperator(const Mesh& m,
                               const std::vector<FaceFrame>& frames) {
    const int F = m.nF();
    const int E = (int)m.eInt.size();
    std::vector<Trip> T;
    T.reserve(4 * E);

    for (int r = 0; r < E; ++r) {
        int h  = m.eInt[r].first;
        int fi = m.he[h].face;
        int fj = m.he[m.he[h].twin].face;
        Eigen::Vector3d ev = m.V.row(m.he[h].vertex) - m.V.row(m.heStart(h));
        double eix = ev.dot(frames[fi].e1);
        double eiy = ev.dot(frames[fi].e2);
        double ejx = ev.dot(frames[fj].e1);
        double ejy = ev.dot(frames[fj].e2);

        T.emplace_back(r, 2*fi,      eix);
        T.emplace_back(r, 2*fi + 1,  eiy);
        T.emplace_back(r, 2*fj,     -ejx);
        T.emplace_back(r, 2*fj + 1, -ejy);
    }
    SpMat C(E, 2 * F);
    C.setFromTriplets(T.begin(), T.end());
    return C;
}

} // namespace wgf
