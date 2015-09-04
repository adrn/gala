#ifndef __PYX_HAVE__gary__integrate___leapfrog
#define __PYX_HAVE__gary__integrate___leapfrog


/* "gary/integrate/_leapfrog.pyx":21
 * from ..potential.cpotential cimport _CPotential
 * 
 * ctypedef public void (*f_type)(_CPotential, double*, double*)             # <<<<<<<<<<<<<<
 * 
 * cdef public void c_init_leapfrog(f_type func, double t, double *w0):
 */
typedef void (*f_type)(struct __pyx_obj_4gary_9potential_10cpotential__CPotential *, double *, double *);

#ifndef __PYX_HAVE_API__gary__integrate___leapfrog

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(void) c_init_leapfrog(f_type, double, double *);
__PYX_EXTERN_C DL_IMPORT(void) c_leapfrog_run(double *, double *, double, int, double);

#endif /* !__PYX_HAVE_API__gary__integrate___leapfrog */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_leapfrog(void);
#else
PyMODINIT_FUNC PyInit__leapfrog(void);
#endif

#endif /* !__PYX_HAVE__gary__integrate___leapfrog */
