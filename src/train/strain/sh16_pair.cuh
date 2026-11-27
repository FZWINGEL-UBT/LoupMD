// Pair-parity spherical harmonics (l <= 3) and matrix-free Jacobian contractions (LST_PAIR_BASIS=1).
// Exact constants match generated_train.cuh's sh16(ux, uy, uz, rinv, Y, JY):
//   JY[m, k] = (dP_m/du_k - l(m) * Y_m(u) * u_k) * rinv.
// For reverse edge e1 = -e0: Y_m(-u) = (-1)^l Y_m(u), JY_m(-u) = (-1)^(l+1) JY_m(u).
#pragma once

namespace lst {

// Y[0..15] only (no JY buffer computed or stored)
template <typename T>
__device__ __forceinline__ void sh16_y(T ux, T uy, T uz, T* __restrict__ Y) {
    const T xx = ux * ux, yy = uy * uy, zz = uz * uz;
    Y[0] = T(1);
    Y[1] = T(1.7320508075688773) * ux;
    Y[2] = T(1.7320508075688773) * uy;
    Y[3] = T(1.7320508075688773) * uz;
    Y[4] = T(3.8729833462074169) * ux * uz;
    Y[5] = T(3.8729833462074169) * ux * uy;
    Y[6] = -T(1.1180339887498948) * xx + T(2.2360679774997897) * yy - T(1.1180339887498948) * zz;
    Y[7] = T(3.8729833462074169) * uy * uz;
    Y[8] = -T(1.9364916731037084) * xx + T(1.9364916731037084) * zz;
    Y[9] = -T(2.0916500663351889) * xx * ux + T(6.2749501990055666) * ux * zz;
    Y[10] = T(10.246950765959598) * ux * uy * uz;
    Y[11] = -T(1.6201851746019651) * xx * ux + T(6.4807406984078602) * ux * yy - T(1.6201851746019651) * ux * zz;
    Y[12] = -T(3.9686269665968859) * xx * uy + T(2.6457513110645906) * yy * uy - T(3.9686269665968859) * uy * zz;
    Y[13] = -T(1.6201851746019651) * xx * uz + T(6.4807406984078602) * yy * uz - T(1.6201851746019651) * zz * uz;
    Y[14] = -T(5.1234753829797992) * xx * uy + T(5.1234753829797992) * uy * zz;
    Y[15] = -T(6.2749501990055666) * xx * uz + T(2.0916500663351889) * zz * uz;
}

// Matrix-free JY(e)^T g for edge with unit vector (ux, uy, uz) = sgn * u_pair, given pair's YP[0..15] and g[0..15]:
//   (JY^T g)_k = (sum_m g_m dP_m/du_k(u_e) - u_{e,k} * sum_m l(m) * g_m * Y_m(u_e)) * rinv.
template <typename T>
__device__ __forceinline__ void sh16_jyt(T ux, T uy, T uz, T rinv, T sgn, const T* __restrict__ yp,
                                         const T* __restrict__ g, T& ox, T& oy, T& oz) {
    const T xx = ux * ux, yy = uy * uy, zz = uz * uz;
    const T xy = ux * uy, xz = ux * uz, yz = uy * uz;
    const T d1 = g[1] * yp[1] + g[2] * yp[2] + g[3] * yp[3];
    const T d2 = g[4] * yp[4] + g[5] * yp[5] + g[6] * yp[6] + g[7] * yp[7] + g[8] * yp[8];
    const T d3 = g[9] * yp[9] + g[10] * yp[10] + g[11] * yp[11] + g[12] * yp[12] + g[13] * yp[13] + g[14] * yp[14] +
                 g[15] * yp[15];
    const T D = sgn * d1 + T(2) * d2 + T(3) * sgn * d3;

    // dP/du_x
    T dx = T(1.7320508075688773) * g[1]
         + T(3.8729833462074169) * (g[4] * uz + g[5] * uy - g[8] * ux)
         - T(2.2360679774997897) * g[6] * ux
         + T(6.2749501990055666) * g[9] * (zz - xx)
         + T(10.246950765959598) * (g[10] * yz - g[14] * xy)
         + g[11] * (-T(4.8605555238058952) * xx + T(6.4807406984078602) * yy - T(1.6201851746019651) * zz)
         - T(7.9372539331937718) * g[12] * xy
         - T(3.2403703492039301) * g[13] * xz
         - T(12.549900398011133) * g[15] * xz;

    // dP/du_y
    T dy = T(1.7320508075688773) * g[2]
         + T(3.8729833462074169) * (g[5] * ux + g[7] * uz)
         + T(4.4721359549995794) * g[6] * uy
         + T(10.246950765959598) * g[10] * xz
         + T(12.96148139681572) * (g[11] * xy + g[13] * yz)
         + g[12] * (-T(3.9686269665968859) * xx + T(7.9372539331937718) * yy - T(3.9686269665968859) * zz)
         + T(5.1234753829797992) * g[14] * (zz - xx);

    // dP/du_z
    T dz = T(1.7320508075688773) * g[3]
         + T(3.8729833462074169) * (g[4] * ux + g[7] * uy + g[8] * uz)
         - T(2.2360679774997897) * g[6] * uz
         + T(12.549900398011133) * g[9] * xz
         + T(10.246950765959598) * (g[10] * xy + g[14] * yz)
         - T(3.2403703492039301) * g[11] * xz
         - T(7.9372539331937718) * g[12] * yz
         + g[13] * (-T(1.6201851746019651) * xx + T(6.4807406984078602) * yy - T(4.8605555238058952) * zz)
         + T(6.2749501990055666) * g[15] * (zz - xx);

    ox = (dx - D * ux) * rinv;
    oy = (dy - D * uy) * rinv;
    oz = (dz - D * uz) * rinv;
}

// Matrix-free JY(e0) * w for canonical pair edge e0:
//   out[m] = (dP_m/du . w - l(m) * y[m] * b) * rinv, where b = u . w.
template <typename T>
__device__ __forceinline__ void sh16_jyv(T ux, T uy, T uz, T rinv, const T* __restrict__ y, T wx, T wy, T wz, T b,
                                         T* __restrict__ out) {
    const T xx = ux * ux, yy = uy * uy, zz = uz * uz;
    const T xy = ux * uy, xz = ux * uz, yz = uy * uz;
    const T b2 = T(2) * b, b3 = T(3) * b;
    out[0] = T(0);
    out[1] = (T(1.7320508075688773) * wx - y[1] * b) * rinv;
    out[2] = (T(1.7320508075688773) * wy - y[2] * b) * rinv;
    out[3] = (T(1.7320508075688773) * wz - y[3] * b) * rinv;
    out[4] = (T(3.8729833462074169) * (uz * wx + ux * wz) - y[4] * b2) * rinv;
    out[5] = (T(3.8729833462074169) * (uy * wx + ux * wy) - y[5] * b2) * rinv;
    out[6] = (-T(2.2360679774997897) * ux * wx + T(4.4721359549995794) * uy * wy - T(2.2360679774997897) * uz * wz -
              y[6] * b2) * rinv;
    out[7] = (T(3.8729833462074169) * (uz * wy + uy * wz) - y[7] * b2) * rinv;
    out[8] = (T(3.8729833462074169) * (uz * wz - ux * wx) - y[8] * b2) * rinv;
    out[9] = (T(6.2749501990055666) * (zz - xx) * wx + T(12.549900398011133) * xz * wz - y[9] * b3) * rinv;
    out[10] = (T(10.246950765959598) * (yz * wx + xz * wy + xy * wz) - y[10] * b3) * rinv;
    out[11] = ((-T(4.8605555238058952) * xx + T(6.4807406984078602) * yy - T(1.6201851746019651) * zz) * wx +
               T(12.96148139681572) * xy * wy - T(3.2403703492039301) * xz * wz - y[11] * b3) * rinv;
    out[12] = (-T(7.9372539331937718) * xy * wx +
               (-T(3.9686269665968859) * xx + T(7.9372539331937718) * yy - T(3.9686269665968859) * zz) * wy -
               T(7.9372539331937718) * yz * wz - y[12] * b3) * rinv;
    out[13] = (-T(3.2403703492039301) * xz * wx + T(12.96148139681572) * yz * wy +
               (-T(1.6201851746019651) * xx + T(6.4807406984078602) * yy - T(4.8605555238058952) * zz) * wz -
               y[13] * b3) * rinv;
    out[14] = (-T(10.246950765959598) * xy * wx + T(5.1234753829797992) * (zz - xx) * wy +
               T(10.246950765959598) * yz * wz - y[14] * b3) * rinv;
    out[15] = (-T(12.549900398011133) * xz * wx + T(6.2749501990055666) * (zz - xx) * wz - y[15] * b3) * rinv;
}

}  // namespace lst
