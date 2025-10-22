#include <math.h>

static inline double norm2(const double *q) {
    return sqrt(q[0]*q[0] + q[1]*q[1]);
}

static inline double norm3(const double *q) {
    return sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
}
