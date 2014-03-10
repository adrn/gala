#include "SgrCoord.h"
#include <fstream>

int main()
{
    using namespace std;

    double l, b, r;
    double Xs, Ys, Zs, lambda, beta, lambda_gc, beta_gc, d;
    double Xsun = 8.0;

    r = 1.0;
    double ls[] = {111.413, 174.123, 18.34, 272.435, 14.341, 1.0, 71.45, 50.13, 200.14, 310.124};
    double bs[] = {13.51, 10.12, -19.1245, 68.46, 45.136, 81.512, -71.235, 21.535, 1.641, -11.51346};

    ofstream output_file;
    output_file.open ("SgrCoord_data");

    output_file << "# l,b,lambda,beta\n";
    for (int i=0; i < 10; i++) {
        char buffer [50];

        l = ls[i];
        b = bs[i];
        LBRtoSgr(l,b,r,Xs,Ys,Zs,lambda,beta,Xsun);

        sprintf(buffer, "%f,%f,%f,%f\n", l, b, lambda, beta);
        output_file << buffer;
    }

    output_file.close();

    return 0;
}
