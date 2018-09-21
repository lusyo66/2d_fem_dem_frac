//
//  el_md_g1int.cpp
//  Jensen_code
//
//  Created by Christopher Kung on 1/20/16.
//  Copyright © 2016 Christopher Kung. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "armadillo"
#include "routines.h"

using namespace arma;
mat el_md_g1int(vec coordsx, vec params) {
    
    double x1, x2, x3, x4, y1, y2, y3, y4;
    double rho, numips;
    double const0, xi, eta;
    double dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi;
    double dN1_deta, dN2_deta, dN3_deta, dN4_deta;
    double dx_dxi, dx_deta, dy_dxi, dy_deta;
    double j;
    int ii;
    
    mat m_e;
    mat xi_vect;
    mat N;
    mat Je;
//    vec xi_vect(2);
    vec weight(4);

    
    x1 = coordsx(0);
    x2 = coordsx(1);
    x3 = coordsx(2);
    x4 = coordsx(3);
    y1 = coordsx(4);
    y2 = coordsx(5);
    y3 = coordsx(6);
    y4 = coordsx(7);
    
    
    rho = params(2);
    
    numips = round(params(4));
    
    
    // Mass Matrix initialization
    m_e.zeros(8,8);
    xi_vect.zeros(4,2);
    N.zeros(2,8);
    Je.zeros(2,2);
    
    // Set Gauss point coordinates in xi space
    const0 = 1.0/sqrt(3.0);
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
    
    for(ii = 0; ii < int(numips); ii++) {
        // Current Gauss point location
        xi = xi_vect(ii,0);
	eta = xi_vect(ii,1);
        
        // shape function vector;
	N(0,1) = N(0,3) = N(0,5) = N(0,7) = N(1,0) = N(1,2) = N(1,4) = N(1,6) = 0.0;
	N(0,0) = N(1,1) = 0.25*(1.0-xi)*(1.0-eta);
	N(0,2) = N(1,3) = 0.25*(1.0+xi)*(1.0-eta);
	N(0,4) = N(1,5) = 0.25*(1.0+xi)*(1.0+eta);	
	N(0,6) = N(1,7) = 0.25*(1.0-xi)*(1.0+eta);

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
        
        m_e = m_e + (N.t()*N) * rho * j * weight(ii);
    }
    
    return m_e;
}
