//
//  el_stress_ellip3d.cpp
//  Jensen_code
//
//  Created by Christopher Kung on 2/3/16.
//  Copyright �� 2016 Christopher Kung. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <string>
#include "armadillo"
#include "routines.h"

using namespace arma;

void el_stress_ellip3d(const char * outputDir,
                       rowvec coordsx,
                       rowvec d,
                       rowvec d_last,
                       vec params,
                       int n_save,
                       int n_stop,
                       int n,
                       int el,
                       int ip,
		       mat D,
                       cube & stress_el,
                       cube & strain_el,
                       double dt,
                       demInput demParams) {
    
    double x1, x2, x3, x4, y1, y2, y3, y4; 
    double numips, nstress, nel;
    double const0, xi, eta;
    double dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi;
    double dN1_deta, dN2_deta, dN3_deta, dN4_deta;
    double dx_dxi, dx_deta, dy_dxi, dy_deta;
    double h0, sig11, sig33, sig13, sig31, trash;
    double shearAngle, shearAngle_n;
    double j;
    double shearRate;
    double dtop_n, dtop, dbot;
    int ii;
    int num_threads;
    
    string dirName;
    
    string::size_type sz, addin;

    mat stress;
    mat strain;
    mat xi_vect;
    mat Je, Jeinv;
    mat B;
    mat dN1, dN2, dN3, dN4;
    mat dN1_dx_vect, dN2_dx_vect, dN3_dx_vect, dN4_dx_vect;
    
    vec f_e;
    vec dudX, dudX_n;
    vec weight(4);
    vec S;
    
    char cCurrentPath[FILENAME_MAX];
    
    getcwd(cCurrentPath, sizeof(cCurrentPath));


#ifdef OPENMP
#pragma omp parallel
    num_threads = omp_get_num_threads();
#pragma end omp parallel

    cout << "Number of threads = " << num_threads << endl;
#endif

    

    sig11 = -1.0;
    sig33 = -1.0;
    sig13 = -1.0;
    sig31 = -1.0;
    sz = -1;
   
    x1 = coordsx(0);
    x2 = coordsx(1);
    x3 = coordsx(2);
    x4 = coordsx(3);
    y1 = coordsx(4);
    y2 = coordsx(5);
    y3 = coordsx(6);
    y4 = coordsx(7); 
    
    numips = round(params(4));
    nstress = round(params(5));
    h0 = params(8); 
    nel = params(9);
    
    // Initialize stress at Gauss points
    stress_el.zeros(numips,nstress,nel);
    strain_el.zeros(numips,nstress,nel);
    stress.zeros(numips, nstress);
    strain.zeros(numips, nstress);
    xi_vect.zeros(4,2);
    Je.zeros(2,2);
    dN1.zeros(1,2);
    dN2.zeros(1,2);
    dN3.zeros(1,2);
    dN4.zeros(1,2);
    B.zeros(3,8);
    dN1_dx_vect.zeros(1,2);
    dN2_dx_vect.zeros(1,2);
    dN3_dx_vect.zeros(1,2);
    dN4_dx_vect.zeros(1,2);
    
    const0 = sqrt(1.0/3.0);
    xi_vect(0,0) = -const0;
    xi_vect(0,1) = -const0;
    xi_vect(1,0) = const0;
    xi_vect(1,1) = -const0;
    xi_vect(2,0) = const0;
    xi_vect(2,1) = const0;
    xi_vect(3,0) = -const0;
    xi_vect(3,1) = const0;
    weight(0) = 1.0;
    weight(1) = 1.0;
    weight(2) = 1.0;
    weight(3) = 1.0;       
 
    cout << "el_stress_ellip3d: Iteration " << ip+1 << " of " << numips << endl;
        
    xi = xi_vect(ip,0);
    eta = xi_vect(ip,1);
    
    // derivatives of shape function with respect to xi
        dN1_dxi = -0.25*(1-eta);
        dN2_dxi = -dN1_dxi;
        dN3_dxi = 0.25*(1+eta);
        dN4_dxi = -dN3_dxi;

        // derivatives of shape function with respect to eta
        dN1_deta = -0.25*(1-xi);
        dN2_deta = -0.25*(1+xi);
        dN3_deta = -dN2_deta;
        dN4_deta = -dN1_deta;

        // calculate jacobian
        dx_dxi = dN1_dxi*x1 + dN2_dxi*x2 + dN3_dxi*x3 + dN4_dxi*x4;
        dx_deta = dN1_deta*x1 + dN2_deta*x2 + dN3_deta*x3 + dN4_deta*x4;
        dy_dxi = dN1_dxi*y1 + dN2_dxi*y2 + dN3_dxi*y3 + dN4_dxi*y4;
        dy_deta = dN1_deta*y1 + dN2_deta*y2 + dN3_deta*y3 + dN4_deta*y4;
        Je(0,0) = dx_dxi;
        Je(0,1) = dx_deta;
        Je(1,0) = dy_dxi;
        Je(1,1) = dy_deta;
        j = dx_dxi*dy_deta - dx_deta*dy_dxi;

        Jeinv=inv(Je);

        dN1(0,0)=dN1_dxi;
        dN1(0,1)=dN1_deta;
        dN2(0,0)=dN2_dxi;
        dN2(0,1)=dN2_deta;
        dN3(0,0)=dN3_dxi;
        dN3(0,1)=dN3_deta;
        dN4(0,0)=dN4_dxi;
        dN4(0,1)=dN4_deta;

        dN1_dx_vect = dN1 * Jeinv;
        dN2_dx_vect = dN2 * Jeinv;
        dN3_dx_vect = dN3 * Jeinv;
        dN4_dx_vect = dN4 * Jeinv;

        B(0,0)=B(2,1)=dN1_dx_vect(0,0);
        B(1,1)=B(2,0)=dN1_dx_vect(0,1);
        B(0,1)=B(1,0)=0;
        B(0,2)=B(2,3)=dN2_dx_vect(0,0);
        B(1,3)=B(2,2)=dN2_dx_vect(0,1); 
        B(0,3)=B(1,2)=0;
        B(0,4)=B(2,5)=dN3_dx_vect(0,0);
        B(1,5)=B(2,4)=dN3_dx_vect(0,1);
        B(0,5)=B(1,4)=0;
        B(0,6)=B(2,7)=dN4_dx_vect(0,0);
        B(1,7)=B(2,6)=dN4_dx_vect(0,1);
        B(0,7)=B(1,6)=0;
    dudX = B*d.t();
    S = D*B*d.t();
    strain.row(ip) = dudX.t();
    stress.row(ip) = S.t();
    strain_el.slice(el)=strain;
    stress_el.slice(el)=stress;    

    dudX_n = B*d_last.t();

        
    dtop_n = h0 * (-dudX_n(1));
    dtop  = h0 * (-dudX(1)) - dtop_n;
    shearAngle_n = atan(dudX_n(2));
    shearAngle = atan(dudX(2))-shearAngle_n;
    
    dirName = string(outputDir) + "/el"+ to_string(el+1) + "_ip" + to_string(ip+1);
        
    cout << "dirName = " << dirName << endl;
        
    // Change to the output directory path so that ellips3d can run
    chdir(dirName.c_str());

    if (demParams.isFrac == 0){
        main_ellip3d(float(dtop), float(shearAngle), n_save, num_threads, dirName, dt, demParams);
    } else if (demParams.isFrac == 1) {
        main_ellip3d_frac(float(dtop), float(dbot), n_save, num_threads, dirName, dt, demParams);
    }
        
    // Change back to the directory from where this code is being run so that file inputs can be referenced properly
    chdir(cCurrentPath);

    //Routines to look at comp_progress
    std:ifstream infile(string(outputDir) + "/el"+ to_string(el+1) + "_ip" + to_string(ip+1) + "/comp_progress");

    //This routine retrieves the last line of comp_progress
    string lastLine = getLastLine(infile);

    // This loop goes through the last line of comp_progress and parses out the approrpriate parts
    for(ii = 0; ii < 130; ii++) {
        if(ii == 0)
            trash = stod(lastLine, &sz);
        else {
            trash = stod(lastLine.substr(sz), &addin);
            sz = sz + addin;
        }
            
        if(ii == 17) sig11 = trash;
	if(ii == 25) sig33 = trash;
	if(ii == 19) sig13 = trash;
	if(ii == 23) sig31 = trash;
    }

        
        
    infile.close();
}
