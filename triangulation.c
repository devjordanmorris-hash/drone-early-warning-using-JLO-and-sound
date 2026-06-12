/**
 * triangulation.c – Barebones 3D triangulation using TDOA
 *
 * MIT License – Copyright (c) 2026 Jordan Morris
 *
 * This file implements a simple Newton‑Raphson solver for
 * acoustic triangulation with 4 microphones.
 *
 * Compile with: gcc -O2 -lm triangulation.c -o triangulation
 * Run: ./triangulation
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SOUND_SPEED 343.0   // m/s at 20°C
#define MAX_ITER 20
#define EPSILON 1e-6

// Microphone positions (metres)
// Square array: M1(0,0,0), M2(w,0,0), M3(0,h,0), M4(w,h,0)
typedef struct {
    double x, y, z;
} Vec3;

// Source position
typedef Vec3 Source;

// Compute Euclidean distance between two points
double distance(const Vec3* a, const Vec3* b) {
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    double dz = a->z - b->z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

// Residual (error) between measured and predicted distance differences
void residuals(const Source* src, const Vec3 mics[4], const double tdoa[3],
               double res[3], double c) {
    // Measured distance differences from TDOA
    double d12_meas = c * tdoa[0];
    double d13_meas = c * tdoa[1];
    double d14_meas = c * tdoa[2];

    // Predicted distance differences from current source estimate
    double d12_pred = distance(src, &mics[0]) - distance(src, &mics[1]);
    double d13_pred = distance(src, &mics[0]) - distance(src, &mics[2]);
    double d14_pred = distance(src, &mics[0]) - distance(src, &mics[3]);

    res[0] = d12_meas - d12_pred;
    res[1] = d13_meas - d13_pred;
    res[2] = d14_meas - d14_pred;
}

// Simple 3x3 matrix inverse (for Newton step)
int invert_3x3(double A[3][3], double inv[3][3]) {
    double det = A[0][0] * (A[1][1]*A[2][2] - A[1][2]*A[2][1])
               - A[0][1] * (A[1][0]*A[2][2] - A[1][2]*A[2][0])
               + A[0][2] * (A[1][0]*A[2][1] - A[1][1]*A[2][0]);

    if (fabs(det) < 1e-12) return 0; // singular

    double idet = 1.0 / det;

    inv[0][0] = (A[1][1]*A[2][2] - A[1][2]*A[2][1]) * idet;
    inv[0][1] = (A[0][2]*A[2][1] - A[0][1]*A[2][2]) * idet;
    inv[0][2] = (A[0][1]*A[1][2] - A[0][2]*A[1][1]) * idet;

    inv[1][0] = (A[1][2]*A[2][0] - A[1][0]*A[2][2]) * idet;
    inv[1][1] = (A[0][0]*A[2][2] - A[0][2]*A[2][0]) * idet;
    inv[1][2] = (A[0][2]*A[1][0] - A[0][0]*A[1][2]) * idet;

    inv[2][0] = (A[1][0]*A[2][1] - A[1][1]*A[2][0]) * idet;
    inv[2][1] = (A[0][1]*A[2][0] - A[0][0]*A[2][1]) * idet;
    inv[2][2] = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) * idet;

    return 1;
}

// Newton‑Raphson solver: find source position given TDOAs and mic positions
int triangulate(const Vec3 mics[4], const double tdoa[3],
                Source* src, int max_iter) {
    // Initial guess: centre of microphones
    src->x = (mics[0].x + mics[1].x + mics[2].x + mics[3].x) / 4.0;
    src->y = (mics[0].y + mics[1].y + mics[2].y + mics[3].y) / 4.0;
    src->z = (mics[0].z + mics[1].z + mics[2].z + mics[3].z) / 4.0;

    double c = SOUND_SPEED;

    for (int iter = 0; iter < max_iter; iter++) {
        double res[3];
        residuals(src, mics, tdoa, res, c);

        // Check convergence
        if (fabs(res[0]) + fabs(res[1]) + fabs(res[2]) < EPSILON) {
            return 1; // converged
        }

        // Compute Jacobian numerically
        double J[3][3];
        double eps = 0.01; // 1 cm perturbation

        for (int d = 0; d < 3; d++) {
            Source src_plus = *src;
            ((double*)&src_plus)[d] += eps;

            double res_plus[3];
            residuals(&src_plus, mics, tdoa, res_plus, c);

            for (int r = 0; r < 3; r++) {
                J[r][d] = (res_plus[r] - res[r]) / eps;
            }
        }

        double invJ[3][3];
        if (!invert_3x3(J, invJ)) {
            return 0; // singular, can't solve
        }

        // Update: src = src + invJ * res
        double dx = invJ[0][0]*res[0] + invJ[0][1]*res[1] + invJ[0][2]*res[2];
        double dy = invJ[1][0]*res[0] + invJ[1][1]*res[1] + invJ[1][2]*res[2];
        double dz = invJ[2][0]*res[0] + invJ[2][1]*res[1] + invJ[2][2]*res[2];

        src->x += dx;
        src->y += dy;
        src->z += dz;
    }

    return 0; // not converged
}

// Convert source position to bearing (degrees from north) and distance
void bearing_distance(const Source* src, const Vec3* reference,
                      double* bearing_deg, double* distance_m) {
    double dx = src->x - reference->x;
    double dy = src->y - reference->y;
    *distance_m = sqrt(dx*dx + dy*dy);
    *bearing_deg = atan2(dx, dy) * 180.0 / M_PI;
    if (*bearing_deg < 0) *bearing_deg += 360.0;
}

// -------------------------------------------------------------------
// Example usage with simulated TDOA values
// -------------------------------------------------------------------
int main() {
    // Microphone positions (square, 0.5 m apart)
    Vec3 mics[4] = {
        {0.0, 0.0, 0.0},   // M1
        {0.5, 0.0, 0.0},   // M2
        {0.0, 0.5, 0.0},   // M3
        {0.5, 0.5, 0.0}    // M4
    };

    // Simulated source location (e.g., drone at (2, 3, 1) metres)
    Source true_src = {2.0, 3.0, 1.0};

    // Compute exact TDOAs from true source (later these will come from JLO)
    double d1 = distance(&true_src, &mics[0]);
    double d2 = distance(&true_src, &mics[1]);
    double d3 = distance(&true_src, &mics[2]);
    double d4 = distance(&true_src, &mics[3]);

    double tdoa[3] = {
        (d1 - d2) / SOUND_SPEED,
        (d1 - d3) / SOUND_SPEED,
        (d1 - d4) / SOUND_SPEED
    };

    printf("True source: (%.2f, %.2f, %.2f)\n", true_src.x, true_src.y, true_src.z);
    printf("TDOA (us): %.2f, %.2f, %.2f\n",
           tdoa[0]*1e6, tdoa[1]*1e6, tdoa[2]*1e6);

    Source estimated;
    int success = triangulate(mics, tdoa, &estimated, MAX_ITER);

    if (success) {
        printf("Estimated source: (%.3f, %.3f, %.3f)\n",
               estimated.x, estimated.y, estimated.z);

        double err = distance(&estimated, &true_src);
        printf("Position error: %.3f m\n", err);

        double bearing, dist;
        bearing_distance(&estimated, &mics[0], &bearing, &dist);
        printf("Bearing from M1: %.1f°, distance: %.2f m\n", bearing, dist);
    } else {
        printf("Triangulation did not converge.\n");
    }

    return 0;
}
