//
//  meshTools.cpp
//  Jensen_code
//
//  Created by Erik Jensen 8/25/2017.
//  Copyright �� 2017 Erik Jensen. All rights reserved.
//

#include "meshTools.h"

using namespace std;
using namespace arma;

void createCoords(mat & coords, vec params, double el_width, double el_height)
{
	int nel = params(9);
	int neldof = params(10);

	coords.set_size(nel,neldof);


	for (int el = 0; el < nel; el++)
	{
	    coords(el,0) = coords(el,3) = 0;
	    coords(el,1) = coords(el,2) = el_width;
	    coords(el,4) = coords(el,5) = el*el_height;
	    coords(el,6) = coords(el,7) = (el+1)*el_height;
	}

}

void createLM(umat & LM, vec params)
{
    int nel = params(9);
	int neldof = params(10);
	LM.set_size(neldof,nel);

	LM(0,0)=LM(2,0)=1;
	LM(1,0)=LM(3,0)=0;
	LM(4,0)=LM(6,0)=2;
        LM(5,0)=LM(7,0)=3;

	for (int el = 1; el < nel; el++)
	{
		LM(0,el) = LM(2,el) = 2*(el+1)-2;
		LM(1,el) = LM(3,el) = 2*(el+1)-1;
		LM(4,el) = LM(6,el) = 2*(el+1);
		LM(5,el) = LM(7,el) = 2*(el+1)+1;
	}
}

void createG(mat & g, vec disp, vec params, int n)
{

	g.zeros(params(10),params(9));
	g(params(10)-1,params(9)-1) = disp(n);

}

void printMesh(mat coords, umat LM, mat g)
{
	cout << endl << endl;
 	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
  	cout << "SUMMARY OF MESH:"                          << endl;
  	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
  	cout << endl;
  	cout << "coords = " 								<< endl;
  	coords.print();
    cout << endl;
  	cout << "LM = " 									<< endl;
  	LM.print();
  	cout << endl;
  	cout << "g = " 										<< endl;
  	g.print();  
  	cout << endl;	
  	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << endl << endl << endl;
}

void whichELIP(int rank, int & el, int & ip, int nel, int numips)
{

	el = rank / numips;

	ip = rank % numips;

	cout<<rank<<' '<<el<<' '<<ip<<endl;
}

void printELIP(int rank, int el, int ip)
{
	
	cout << endl;
	cout << "rank = " << rank << " :: el = " << el << " :: ip = " << ip;
	cout << endl;

}

