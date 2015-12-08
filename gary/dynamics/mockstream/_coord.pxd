cdef void sat_rotation_matrix(double *w, double *R)

cdef void to_sat_coords(double *w, double *R,
                        double *w_prime)

cdef void from_sat_coords(double *w_prime, double *R,
                          double *w)

cdef void car_to_cyl(double *w, double *cyl)
cdef void cyl_to_car(double *cyl, double *w)
