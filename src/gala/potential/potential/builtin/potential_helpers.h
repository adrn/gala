#include <math.h>
#include "src/vectorization.h"

static inline double norm2_sq(const double *q) {
    return q[0]*q[0] + q[1]*q[1];
}

static inline double norm2(const double *q) {
    return sqrt(norm2_sq(q));
}

static inline double norm3_sq(const double *q) {
    return q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
}

static inline double norm3(const double *q) {
    return sqrt(norm3_sq(q));
}

static inline double norm3_flat_z(const double *q, const double qz) {
    return sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]/(qz*qz));
}

// Helper functions for computing norms with double6ptr
static inline double norm3_sq(const double6ptr& q) {
    return (*q.x) * (*q.x) + (*q.y) * (*q.y) + (*q.z) * (*q.z);
}

static inline double norm3(const double6ptr& q) {
    return sqrt(norm3_sq(q));
}

static inline double norm3_flat_z(const double6ptr& q, const double qz) {
    return sqrt((*q.x) * (*q.x) + (*q.y) * (*q.y) + (*q.z) * (*q.z) / (qz * qz));
}
