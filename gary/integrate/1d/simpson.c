#include <stdio.h>
#include <math.h>

// Simpson's rule
double _simpson (double* y, double dx, int n) {
    /*
        Only works on a fixed grid (e.g., with fixed stepsize in x, dx).
    */

    int i;
    double sum1=0., sum2=0., sum=0.;

    if ((n % 2) == 0) {
        // average the values obtained from doing trapezoidal rule for first interval,
        // simpsons rule for the rest vs. simpsons rule for first (N-2) intervals,
        // then trapezoid rule for the last.
        n--;

        for (i=0; i<n-1; i+=2) {
            sum1 += y[i] + 4*y[i+1] + y[i+2];
        }
        sum1 *= dx/3.;
        sum1 += dx/2. * (y[n-1] + y[n]);

        for (i=1; i<n; i+=2) {
            sum2 += y[i] + 4*y[i+1] + y[i+2];
        }
        sum2 *= dx/3.;
        sum2 += dx/2. * (y[0] + y[1]);
        sum = (sum1 + sum2) / 2.;

    } else {
        for (i=0; i<n-1; i+=2) {
            sum += y[i] + 4*y[i+1] + y[i+2];
        }
        sum *= dx / 3.;
    }

    return sum;
}

// /* Integration using Gauss' rule */
// float gaussint (int no, float min, float max)
// {
//     int n;
//     float quadra = 0.;
//     double w[1000], x[1000];                 /* for points and weights */

//     gauss (no, 0, min, max, x, w);         /* returns Legendre */
//                            /* points and weights */
//     for (n=0; n< no; n++)
//     {
//         quadra += f(x[n])*w[n];                /* calculating the integral */
//     }
//     return (quadra);
// }
