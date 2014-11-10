// SgrCoord.h
// C++ header file of transformation code to the Sgr longitudinal coordinate systems
// defined by Majewski et al. 2003 (ApJ, 599, 1082).
// Author: David R. Law (drlaw@virginia.edu), University of Virginia
// June 2003
// http://www.astro.virginia.edu/~drl5n/Sgr/
//
// This transformation code has been made publically available to promote the use
// of the Sgr longitudinal coordinate system, and may be used freely.  However,
// please acknowledge this website when using this code, and leave all
// header information intact.
//
// Last modified Jan 2010.
// Modification revises Z_Sgr and Z_Sgr,GC to be positive in the direction of the
// orbital pole of Sgr (i.e., match the convention already used for beta)

#include <cmath>
#include <fstream>
using namespace std;

// Transform positions from standard left handed Galactocentric XYZ to
// the heliocentric Sgr system (lambda=0 at Sgr)
// Input must be in kpc of the form X Y Z
// Output is in kpc and degrees, of the form X_Sgr Y_Sgr Z_Sgr r lambda beta
void XYZtoSgr(double X,double Y,double Z,double &Xs,double &Ys,double &Zs,double &r,double &lambda,double &beta,double Xsun=7.0)  
  {
  double radpdeg=3.141592653589793/180.;
  // Define the Euler angles
  double phi=(180+3.75)*radpdeg;
  double theta=(90-13.46)*radpdeg;
  double psi=(180+14.111534)*radpdeg;

  // Define the rotation matrix from the Euler angles
  double rot11=cos(psi)*cos(phi)-cos(theta)*sin(phi)*sin(psi);
  double rot12=cos(psi)*sin(phi)+cos(theta)*cos(phi)*sin(psi);
  double rot13=sin(psi)*sin(theta);
  double rot21=-sin(psi)*cos(phi)-cos(theta)*sin(phi)*cos(psi);
  double rot22=-sin(psi)*sin(phi)+cos(theta)*cos(phi)*cos(psi);
  double rot23=cos(psi)*sin(theta);
  double rot31=sin(theta)*sin(phi);
  double rot32=-sin(theta)*cos(phi);
  double rot33=cos(theta);

  X=-X; // Make the input system right-handed
  X=X+Xsun; // Transform the input system to heliocentic right handed coordinates
  
  // Calculate X,Y,Z,distance in the Sgr system
  Xs=rot11*X+rot12*Y+rot13*Z;
  Ys=rot21*X+rot22*Y+rot23*Z;
  Zs=rot31*X+rot32*Y+rot33*Z;
  r=sqrt(Xs*Xs+Ys*Ys+Zs*Zs);

  Zs=-Zs;
  // Calculate the angular coordinates lambda,beta
  lambda=atan2(Ys,Xs)/radpdeg;
  if (lambda<0) lambda=lambda+360;
  beta=asin(Zs/sqrt(Xs*Xs+Ys*Ys+Zs*Zs))/radpdeg;
  }


// Transform positions from Galactic coordinates (l,b,r) to
// the heliocentric Sgr system (lambda=0 at Sgr)
// Input must be in degrees and kpc of the form l b r
// Output is in kpc and degrees, of the form X_Sgr Y_Sgr Z_Sgr r lambda beta
void LBRtoSgr(double l,double b,double r,double &Xs,double &Ys,double &Zs,double &lambda,double &beta,double Xsun=7.0)  
  {
  double radpdeg=3.141592653589793/180.;
  double X,Y,Z;

  // Transform l,b to radians
  l=l*radpdeg; b=b*radpdeg;
  
  // Transform to heliocentric Cartesian coordinates
  X=r*cos(b)*cos(l);
  Y=r*cos(b)*sin(l);
  Z=r*sin(b);

  // Transform to Galactocentric left handed frame
  X=-X;
  X=X+Xsun;

  // Transform from left handed Galactocentric to Sgr coordinates
  XYZtoSgr(X,Y,Z,Xs,Ys,Zs,r,lambda,beta,Xsun);
  }


// Transform positions from standard left handed Galactocentric XYZ to
// the Galactocentric Sgr system (lambda=0 at the Galactic plane)
// Input must be in kpc of the form X Y Z
// Output is in kpc and degrees, of the form X_Sgr,GC Y_Sgr,GC Z_Sgr,GC d_GC lambda_GC beta_GC
// Note that d is distance from Galactic Center
void XYZtoSgrGC(double X,double Y,double Z,double &Xs,double &Ys,double &Zs,double &d,double &lambda,double &beta,double Xsun=7.0)  
  {
  double radpdeg=3.141592653589793/180.;
  // Define the Euler angles
  double phi=(180+3.75)*radpdeg;
  double theta=(90-13.46)*radpdeg;
  double psiGC=(180+21.604399)*radpdeg;
  // Rotation angle of phiGC past 180degrees is a useful number
  double ang=21.604399*radpdeg;
  // Note that the plane does not actually include the G.C., although it is close
  double xcenter=-8.5227;
  double ycenter=-.3460;
  double zcenter=-.0828;
  double Temp,Temp2,Temp3;

  // Define the rotation matrix from the Euler angles
  double GCrot11=cos(psiGC)*cos(phi)-cos(theta)*sin(phi)*sin(psiGC);
  double GCrot12=cos(psiGC)*sin(phi)+cos(theta)*cos(phi)*sin(psiGC);
  double GCrot13=sin(psiGC)*sin(theta);
  double GCrot21=-sin(psiGC)*cos(phi)-cos(theta)*sin(phi)*cos(psiGC);
  double GCrot22=-sin(psiGC)*sin(phi)+cos(theta)*cos(phi)*cos(psiGC);
  double GCrot23=cos(psiGC)*sin(theta);
  double GCrot31=sin(theta)*sin(phi);
  double GCrot32=-sin(theta)*cos(phi);
  double GCrot33=cos(theta);

  X=-X; // Make the input system right-handed
  X=X+Xsun; // Transform the input system to heliocentric right handed coordinates

  // Calculate Z,distance in the SgrGC system
  Temp=GCrot11*(X+xcenter)+GCrot12*(Y-ycenter)+GCrot13*(Z-zcenter);
  Temp2=GCrot21*(X+xcenter)+GCrot22*(Y-ycenter)+GCrot23*(Z-zcenter);
  Zs=GCrot31*(X+xcenter)+GCrot32*(Y-ycenter)+GCrot33*(Z-zcenter);
  d=sqrt(Temp*Temp+Temp2*Temp2+Zs*Zs);

  Zs=-Zs;
  // Calculate the angular coordinates lambdaGC,betaGC
  Temp3=atan2(Temp2,Temp)/radpdeg;
  if (Temp3<0) Temp3=Temp3+360;
  Temp3=Temp3+ang/radpdeg;
  if (Temp3>360) Temp3=Temp3-360;
  lambda=Temp3;
  beta=asin(Zs/sqrt(Temp*Temp+Temp2*Temp2+Zs*Zs))/radpdeg;

  // Calculate X,Y in the SgrGC system
  Xs=Temp*cos(ang)-Temp2*sin(ang);
  Ys=Temp*sin(ang)+Temp2*cos(ang);
  }


// Transform positions from Galactic coordinates (l,b,r) to
// the Galactocentric Sgr system (lambda=0 at the Galactic plane)
// Input must be in degrees and kpc of the form l b r
// Output is in kpc and degrees, of the form X_Sgr,GC Y_Sgr,GC Z_Sgr,GC d_GC lambda_GC beta_GC
// Note that d is distance from Galactic Center
void LBRtoSgrGC(double l,double b,double r,double &Xs,double &Ys,double &Zs,double &d,double &lambda,double &beta,double Xsun=7.0)  
  {
  double radpdeg=3.141592653589793/180.;
  double X,Y,Z;

  // Transform l,b to radians
  l=l*radpdeg; b=b*radpdeg;
  
  // Transform to heliocentric Cartesian coordinates
  X=r*cos(b)*cos(l);
  Y=r*cos(b)*sin(l);
  Z=r*sin(b);

  // Transform to Galactocentric left handed frame
  X=-X;
  X=X+Xsun;

  // Transform from left handed Galactocentric to Sgr coordinates
  XYZtoSgrGC(X,Y,Z,Xs,Ys,Zs,d,lambda,beta);
  }
