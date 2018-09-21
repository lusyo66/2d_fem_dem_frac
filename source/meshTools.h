//
//  meshTools.cpp
//  Jensen_code
//
//  Created by Erik Jensen 8/25/2017.
//  Copyright �� 2017 Erik Jensen. All rights reserved.
//

#include "armadillo"

using namespace arma;

void createCoords(mat & coords, vec params, double el_width, double el_height);

void createLM(umat & LM, vec params);

void createG(mat & g, vec disp, vec params, int n);

void printMesh(mat coords, umat LM, mat g);

void whichELIP(int rank, int & el, int & ip, int nel, int numips);

void printELIP(int rank, int el, int ip);
