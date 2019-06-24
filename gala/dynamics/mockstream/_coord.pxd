# cython: language_level=3

# cdef void cross(double[::1] x, double[::1] y, double[::1] z)
cdef void cross(double *x, double *y, double *z)
cdef double norm(double *x, int n)
cdef void apply_3matrix(double[:, ::1] R, double *x, double *y, int transpose)

cdef void sat_rotation_matrix(double *w, double *R)

cdef void to_sat_coords(double *w, double *R,
                        double *w_prime)

cdef void from_sat_coords(double *w_prime, double *R,
                          double *w)

cdef void car_to_cyl(double *w, double *cyl)
cdef void cyl_to_car(double *cyl, double *w)
