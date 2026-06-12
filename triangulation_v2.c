/**
 * triangulation_v2.c – Stable TDOA triangulation with LM damping
 *
 * Compile: gcc -O2 -lm triangulation_v2.c -o triangulation_v2
 * Run: ./triangulation_v2
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SOUND_SPEED 343.0
#define MAX_ITER 100
#define EPSILON 1e-8
#define LM_LAMBDA_INIT 0.01   // Levenberg–Marquardt damping factor

typedef struct { double x, y, z; } Vec3;
typedef Vec3 Source;

double distance(const Vec3* a, const Vec3* b) {
    double dx = a->x - b->x, dy = a->y - b->y, dz = a->z - b->z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

// Residuals (error) as a single vector of length 3
void residuals(const Source* src, const Vec3 mics[4], const double tdoa[3],
               double res[3], double c) {
    double d12_meas = c * tdoa[0];
    double d13_meas = c * tdoa[1];
    double d14_meas = c * tdoa[2];

    double d12_pred = distance(src, &mics[0]) - distance(src, &mics[1]);
    double d13_pred = distance(src, &mics[0]) - distance(src, &mics[2]);
    double d14_pred = distance(src, &mics[0]) - distance(src, &mics[3]);

    res[0] = d12_meas - d12_pred;
    res[1] = d13_meas - d13_pred;
    res[2] = d14_meas - d14_pred;
}

// Compute cost = sum(res^2)
double cost(const Source* src, const Vec3 mics[4], const double tdoa[3], double c) {
    double res[3];
    residuals(src, mics, tdoa, res, c);
    return res[0]*res[0] + res[1]*res[1] + res[2]*res[2];
}

// Levenberg–Marquardt solver for TDOA triangulation
int triangulate_lm(const Vec3 mics[4], const double tdoa[3],
                   Source* src, int max_iter) {
    double c = SOUND_SPEED;
    double lambda = LM_LAMBDA_INIT;

    // Initial guess: centre of array + offset based on TDOA signs
    src->x = (mics[0].x + mics[1].x + mics[2].x + mics[3].x) / 4.0;
    src->y = (mics[0].y + mics[1].y + mics[2].y + mics[3].y) / 4.0;
    src->z = (mics[0].z + mics[1].z + mics[2].z + mics[3].z) / 4.0;

    // Heuristic: shift initial guess towards the direction of the largest TDOA
    // If Δd14 is largest, move towards the mic that received the sound first (M1?)
    // This is a rough but often effective starting point.
    if (tdoa[0] > 0) { src->x += 0.5; } else { src->x -= 0.5; }
    if (tdoa[1] > 0) { src->y += 0.5; } else { src->y -= 0.5; }
    src->z += (tdoa[0] + tdoa[1] + tdoa[2]) * SOUND_SPEED / 3.0;

    double cost_prev = cost(src, mics, tdoa, c);
    printf("Initial guess: (%.2f, %.2f, %.2f), cost = %.6f\n",
           src->x, src->y, src->z, cost_prev);

    for (int iter = 0; iter < max_iter; iter++) {
        // Compute Jacobian numerically
        double J[3][3];
        double eps = 0.01; // 1 cm

        for (int d = 0; d < 3; d++) {
            Source src_plus = *src;
            ((double*)&src_plus)[d] += eps;

            double res_plus[3];
            residuals(&src_plus, mics, tdoa, res_plus, c);

            double res[3];
            residuals(src, mics, tdoa, res, c);

            for (int r = 0; r < 3; r++) {
                J[r][d] = (res_plus[r] - res[r]) / eps;
            }
        }

        // Compute normal equations: (J^T J + λI) Δ = J^T r
        double JTJ[3][3] = {{0}};
        double JTr[3] = {0};

        for (int i = 0; i < 3; i++) {
            double res[3];
            residuals(src, mics, tdoa, res, c);
            for (int j = 0; j < 3; j++) {
                JTJ[j][i] += J[j][i];
                JTr[i] += J[j][i] * res[j];
            }
        }

        // Add damping
        for (int i = 0; i < 3; i++) {
            JTJ[i][i] += lambda;
        }

        // Solve (JTJ) Δ = JTr using Gaussian elimination (3x3)
        double A[3][3], b[3];
        memcpy(A, JTJ, sizeof(A));
        memcpy(b, JTr, sizeof(b));

        // Gaussian elimination
        for (int col = 0; col < 3; col++) {
            double pivot = A[col][col];
            if (fabs(pivot) < 1e-12) {
                // singular – increase lambda and retry
                lambda *= 10;
                col = -1; // restart elimination
                memcpy(A, JTJ, sizeof(A));
                memcpy(b, JTr, sizeof(b));
                for (int i = 0; i < 3; i++) A[i][i] += lambda;
                continue;
            }
            for (int row = col + 1; row < 3; row++) {
                double factor = A[row][col] / pivot;
                for (int k = col; k < 3; k++) {
                    A[row][k] -= factor * A[col][k];
                }
                b[row] -= factor * b[col];
            }
        }

        double dx[3];
        for (int i = 2; i >= 0; i--) {
            dx[i] = b[i];
            for (int j = i + 1; j < 3; j++) {
                dx[i] -= A[i][j] * dx[j];
            }
            dx[i] /= A[i][i];
        }

        // Try the step
        Source new_src = *src;
        new_src.x += dx[0];
        new_src.y += dx[1];
        new_src.z += dx[2];

        double cost_new = cost(&new_src, mics, tdoa, c);

        if (cost_new < cost_prev) {
            // Accept step, reduce lambda
            *src = new_src;
            cost_prev = cost_new;
            lambda *= 0.1;
            if (lambda < 1e-8) lambda = 1e-8;

            if (cost_prev < EPSILON) {
                printf("Converged after %d iterations, cost = %.2e\n", iter + 1, cost_prev);
                return 1;
            }
        } else {
            // Reject step, increase lambda
            lambda *= 10;
            if (lambda > 1e12) {
                printf("Lambda too large, giving up at iteration %d\n", iter);
                return 0;
            }
        }
    }
    return 0;
}

int main() {
    // Microphone positions (square, 0.5 m apart)
    Vec3 mics[4] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0},
        {0.5, 0.5, 0.0}
    };

    // Your TDOA values (microseconds)
    double tdoa_us[3] = {704.54, 1129.90, 1922.60};
    double tdoa[3] = {tdoa_us[0] * 1e-6, tdoa_us[1] * 1e-6, tdoa_us[2] * 1e-6};

    printf("TDOA (us): %.2f, %.2f, %.2f\n", tdoa_us[0], tdoa_us[1], tdoa_us[2]);

    Source estimated;
    int success = triangulate_lm(mics, tdoa, &estimated, MAX_ITER);

    if (success) {
        printf("\nEstimated source: (%.3f, %.3f, %.3f) m\n",
               estimated.x, estimated.y, estimated.z);

        double bearing = atan2(estimated.y, estimated.x) * 180.0 / M_PI;
        double distance = sqrt(estimated.x*estimated.x + estimated.y*estimated.y);
        printf("Bearing: %.1f°, distance: %.2f m\n", bearing, distance);
    } else {
        printf("\nTriangulation did not converge.\n");
        printf("Check that your TDOA values are consistent.\n");
        printf("Try using real data from JLO zero‑crossing detection.\n");
    }

    return 0;
}
