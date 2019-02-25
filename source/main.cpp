//
//  main.cpp
//  Jensen_code
//
//  Created by Christopher Kung on 1/20/16.
//  Copyright �� 2016 Christopher Kung. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <fstream>

#include "armadillo"
#include "routines.h"
#include "meshTools.h"
#include "dispFunctions.h"

#ifdef USE_MPI
#include "mpi.h"
#endif

using namespace std;
using namespace arma;

int main(int argc, char * argv[]) {

    double lambda, mu, rho, grav, nu, Emod, shear_vel;
    double shear_vel_rock, rho_rock;
    double zeta, omega1, omega2, a_R, b_R;
    double fmax, wavelength, nel_wave, el_height, el_width;
    double h_DEM, w_DEM, l_DEM, A_DEM;
    double t, dt, grav_t_ramp, time_tot, nsteps_g, nsteps_F;
    double soil_height;
    double gd_n, gd;
    double c_dashpot;
    double traction_max, traction, F_F;
    double beta, gamma, tolr, tola, iter_break;
    double tract, normR;
    double K_temp, K_total, F_S_temp, F_S_total;
    double grav_factor, cFactor;
    double velHist;
    double omega, vel0;
    
    int numips, nstress, nisv, ndof, nel, neldof, ncoords;
    int nsteps, print_int, n_print;
    int ii, jj, kk, ll, n, el, ip;
    int I, J;
    int Rtol;
    int numtasks, rank, rc;
    
    string dirName, dirPrefix;
    
    mat coords, g, g_n;
    mat D, Fff, FF;
    mat fg_el, fs_el, d_el, d_el_last, v_el, a_el, a_el_last, Delta_a_el, Delta_d_el;
    mat G, M, C, K, K_el;
    mat asolve, dsolve;
    mat stress_el_last, strain_el_last;
    
    cube stress_dem_el, stress_el, mass_el, stiff_el, strain_el, Kk, Mm;
    
    vec params, dispfun_time, dispfun_disp, dispfun_eps;
    vec Ff, Ftotal, RHS, F, F_G, F_S;
    vec d, v, a;
    vec tsolve, FFsolve;
    vec dtilde, vtilde;
    vec temp;
    vec del_a, Delta_a, Delta_d;
    
    rowvec d_last, v_last, a_last;
    vec d_pred, v_pred;
    
    umat LM;
    
    struct stat info;
    
    const char * BI_file_Path, * PI_file_Path, * outputDir, * qdel_Path, * fem_inputs, * dem_inputs;
    
    ofstream d_file, v_file, a_file, t_file, stress_dem_file,  stress_file, strain_file, F_S_file;
    char cCurrentPath[FILENAME_MAX];

#ifdef USE_MPI
    // Initialize MPI
    rc = MPI_Init(&argc, &argv);
    if(rc != MPI_SUCCESS) {
    	cout << "Error starting program with MPI..." << endl;
    	MPI_Abort(MPI_COMM_WORLD, rc);
    }
#endif

    if(argc > 7 || argc < 7) {
        cout << "Input Format: " << argv[0] << " <Path to Boundary Input File> <Path to Particle Input File> "
             << "<Path to qdelauny> <Directory to Write Outputs> <Path to FEM Inputs> <Path to DEM Inputs>" << endl;
        return -1;
    }

#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	cout << "rank = " << rank << endl;
	cout << "numtasks = " << numtasks << endl;
#else
    rank = 0;
    numtasks = 1;
#endif

    BI_file_Path = argv[1];
    PI_file_Path = argv[2];
    qdel_Path    = argv[3];
    outputDir    = argv[4];
    fem_inputs   = argv[5];
    dem_inputs   = argv[6];

    if(rank==0) {
    	cout << "Armadillo version: " << arma_version::as_string() << endl;
    	if(ifstream(BI_file_Path)) {
    		cout << "Boundary Input File found..." << endl;
    	}
    	else {
    		cout << "Boundary Input File not found...exiting..." << endl;
    		return -1;
    	}
    	if(ifstream(PI_file_Path)) {
    		cout << "Particle Input File found..." << endl;
    	}
    	else {
    		cout << "Particle Input File not found...exiting..." << endl;
    		return -1;
    	}
    	if(ifstream(qdel_Path)) {
    		cout << "qdelaunay binary found..." << endl;
    	}
    	else {
    		cout << "qdelaunay binary not found...exiting..." << endl;
    		return -1;
    	}
    	if(ifstream(fem_inputs)) {
    		cout << "FEM input file found..." << endl;
    	}
    	else {
    		cout << "FEM input file not found...exiting..." << endl;
    		return -1;
    	}
        if(ifstream(dem_inputs)) {
            cout << "DEM input file found..." << endl;
        }
        else {
            cout << "DEM input file not found...exiting..." << endl;
            return -1;
        }
    	// Get Current Directory Path where the programming is running
    	if (!getcwd(cCurrentPath, sizeof(cCurrentPath)))
    	{
    		return errno;
    	}

    	cout << "Current Directory is " << cCurrentPath << endl;
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    //Read data from input file (fem_inputs)
    femInput femParams;
    femParams.readData(fem_inputs);

    //Read data from input file (dem_inputs)
    demInput demParams;
    demParams.readData(dem_inputs);

    //Elatic parameters taking from dry mason sand calibration effort
    lambda     = femParams.lambda;  // Pa
    mu         = femParams.mu;      // Pa
    rho        = femParams.rho;     // kg/m^3
    Emod       = 2*mu*(1+nu);	    // Pa
    
    //Gravitational Acceleration
    grav       = femParams.grav;    // m/s^2
       
    //Geometry
    d          = femParams.d;       // m

    // DEM Geometry
    h_DEM      = femParams.h_DEM;         // m
    w_DEM      = femParams.w_DEM;         // m
    l_DEM      = femParams.l_DEM;         // m
    A_DEM      = l_DEM * w_DEM; // m^2
    
    // FEM Constants
    numips     = femParams.numips;
    nstress    = femParams.nstress;
    nisv       = femParams.nisv;
    ndof       = femParams.ndof;
    nel        = femParams.nel;
    neldof     = femParams.neldof;

    // Time Parameters
    t          = femParams.t;
    dt         = femParams.dt;
    print_int  = femParams.print_int;
    n_print    = femParams.n_print;
    time_tot   = femParams.time_tot;
    nsteps     = round(time_tot/dt);

    // Boundary Conditions

    //Damping

    params.set_size(12);
    params(0 ) = lambda;
    params(1 ) = mu;
    params(2 ) = rho;
    params(3 ) = grav;
    params(4 ) = numips;
    params(5 ) = nstress;
    params(6 ) = nisv;
    params(7 ) = 1.0;
    params(8 ) = h_DEM;
    params(9 ) = nel;
    params(10) = neldof;
    params(11) = ndof;

    el_height = 0.25;
    el_width = el_height;

    createCoords(coords,params,el_width,el_height);
    createLM(LM,params);

#ifdef USE_MPI
    whichELIP(rank, el, ip, nel, numips);
#endif

    dispfun_time.zeros(nsteps);
    dispfun_disp.zeros(nsteps);
    dispfun_eps.zeros(nsteps);
/*    
    if (femParams.whichDisp == 1)
    {
        finiteAppliedDisp(dispfun_time, dispfun_disp, dispfun_eps, nsteps, time_tot, strainrate, h);

    } else if (femParams.whichDisp == 2)
    {
        shpbAppliedDisp(dispfun_time, dispfun_disp, dispfun_eps, nsteps, time_tot, strainrate, h);

    } else 
    {
        cout << "ERROR: NO SPECIFICED DISPLACEMENT" << endl;
        exit(0);
    }
*/    
//    createG(g, dispfun_disp, params, 0);
    gd_n = 0.0;
    gd   = 0.0;

    // Concentrated External Force Vector
    traction_max = 0.0;
    traction     = 0.0;
    F_F          = traction*1.0;

    grav_t_ramp = 10.0;
    nsteps_g = round(grav_t_ramp/dt);
    nsteps_F = round(time_tot/dt);
    nsteps = nsteps_g+nsteps_F;

    // Initialization
    g.zeros(neldof,nel);
    stress_dem_el.zeros(numips,nstress,nel);
    stress_el.zeros(numips,nstress,nel);
    strain_el.zeros(numips,nstress,nel);
    stress_el_last.zeros(numips,nstress);
    strain_el_last.zeros(numips,nstress);
    mass_el.zeros(neldof,neldof,nel);
    stiff_el.zeros(neldof,neldof,nel);
    Kk.zeros(neldof,neldof,nel);
    Mm.zeros(neldof,neldof,nel);
    D.zeros(nstress,nstress);
    fg_el.zeros(neldof,nel);
    fs_el.zeros(neldof,nel);
    Fff.zeros(neldof,nel);
    FF.zeros(5,nsteps+1);
    d_el.zeros(nel,neldof);
    v_el.zeros(nel,neldof);
    a_el.zeros(nel,neldof);
    a_el_last.zeros(nel,neldof);
    Delta_a_el.zeros(nel,neldof);
    G.zeros(5,5);
    M.zeros(5,5);
    C.zeros(5,5);
    K.zeros(5,5);
    F.zeros(5);
    F_G.zeros(5);
    F_S.zeros(5);
    Ff.zeros(5);
    Ftotal.zeros(5);

    // Newmark Method Parameters - O(2), explicit
    beta = 0.25;
    gamma = 1.0/2.0;
    

    D(0,0) = D(1,1) = lambda+2*mu;
    D(0,1) = D(1,0) = lambda;
    D(0,2) = D(1,2) = D(2,0) = D(2,1) = 0;
    D(2,2) = mu;

    // Initial Conditions
    // Both M and C are constants

    for(ii = 0; ii < nel; ii++) {
	el_kd_g2int(coords.row(ii), params, ii, D, Kk.slice(ii));
        Mm.slice(ii) = el_md_g1int((coords.row(ii)).t(), params);
        cout<<"internal force"<<endl;
        Fff.col(ii)     = el_f_g4int( (coords.row(ii)).t(), params);
    }

    for(kk = 0; kk < nel; kk++) {
        temp = conv_to<vec>::from(LM.col(kk));
        for(ii = 0; ii < neldof; ii++) {
            I = int(temp(ii));
            if(I > 0) {
                //Note that indices in C++/Armadillo start at 0, not 1 as in Matlab
                Ff(I-1) = Ff(I-1) + Fff(ii,kk);
                for(jj = 0; jj < neldof; jj++) {
                    J = int(temp(jj));
                    if(J > 0) {
                        M(I-1, J-1) = M(I-1, J-1) + Mm(ii,jj,kk);
                        C(I-1, J-1) = C(I-1, J-1) + Kk(ii,jj,kk);
                    }
                }
            }
        }
    }
    cout<<"mass and damping matrix done"<<endl;
    double pi=3.14159265359;
    zeta = 0.0;
    omega1 = 2*pi*0.2;
    omega2 = 2*pi*20;
    a_R = 2*zeta*omega1*omega2/(omega1+omega2);
    b_R = 2*zeta/(omega1+omega2);
    C = a_R*M+b_R*K;
    rho_rock = 2400;
    shear_vel_rock = 760;
    cFactor = (1*el_width)*rho_rock*shear_vel_rock;
    c_dashpot = cFactor;
    C(0,0)=C(0,0)+c_dashpot;
    //Need to add damping to this initialization
    // a = solve(M,-F);
    G = M+gamma*dt*C+beta*dt*dt*K;
    
    //Need to change if ICs are anything other than zero
    d.zeros(5);
    d_last.zeros(5);
    v.zeros(5);
    a.zeros(5);
    
    // Make the necessary directories - only need 1 per IP per EL
    // DEM "snapshot" per call

    if(rank == 0) {

    	for(jj = 1; jj <= nel; jj++) {
    		for(ii = 1; ii <= numips; ii++) {
    			dirName = string(outputDir) + "/el" + to_string(jj) + "_ip" + to_string(ii);
    			cout << "*** Attemping to Creating Directory: " << dirName << endl;
    			if( stat( dirName.c_str(), &info ) != 0 ) {
    				printf( "*** Making directory: %s\n", dirName.c_str() );
    				mkdir(dirName.c_str(),0700);
    			}
    			else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows
    				printf( "*** %s already exists ***\n", dirName.c_str() );
    			else
    				printf( "*** %s is no directory\n", dirName.c_str() );

    			copyfile(BI_file_Path, dirName+"/input_boundary_file");
    			copyfile(PI_file_Path, dirName+"/input_particle_file");

    			// ellip3D is now coupled to this code. no need to precompile versions,
    			// but need to update the fracture version

    			copyfile(qdel_Path, dirName+"/qdelaunay");
    			string qdel_Path = dirName + "/qdelaunay";
    			chmod(qdel_Path.c_str(), S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH);
    		}
    	}

    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (rank == 0)
    {
        femParams.echoData();
        femParams.checkData();
        demParams.echoData();
        demParams.checkData();
        printMesh(coords, LM, g);
    }

    femParams.~femInput();
cout<<"fem input"<<endl;
    grav_t_ramp=0.0;
	nsteps_g=round(grav_t_ramp/dt);
	nsteps_F=round((time_tot/dt));
	nsteps=nsteps_g+nsteps_F;
	asolve.zeros(nsteps+1,5);
	dsolve.zeros(nsteps+1,5);
	tsolve.zeros(nsteps+1);
	FFsolve=zeros(nsteps+1);
	d_el=zeros(nel,neldof);
	Delta_d_el=zeros(nel,neldof);
	d_el_last=zeros(nel,neldof);
	stress_el=zeros(numips, nstress, nel);
	strain_el=zeros(numips, nstress, nel);
	tolr=1e-8;
	tola=1e-8;

 /*   ifstream vinput("/u/yslee/2d_fem_dem/source/velocityHistory.out");
 for (int s; s<7990; s++)
  {
	vinput>>velHist;
	FF(0,nsteps_g-1+s)=cFactor*velHist;
    }
    vinput.close();
*/
	omega = 2*3.14159*2e+4;
	vel0 = 1e+3;
	iter_break=10;
cout<<"Ff"<<endl;
if (rank==0){
    cout<<"Ff"<<endl<<Ff<<endl;
    cout<<"M"<<endl<<M<<endl;
    cout<<"K"<<endl<<K<<endl;
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    for(n = 1; n < nsteps; n++) {

    	if(rank == 0) {
    		cout << "*** Current step = " << n << " ***" << endl;
    	}

#ifdef USE_MPI
    	MPI_Barrier(MPI_COMM_WORLD);
#endif

        t = t + dt;

//        createG(g_n, dispfun_disp, params, n-1);

        if(t < grav_t_ramp) {
            grav_factor = t/grav_t_ramp;
        }
        else {
            grav_factor = 1;
        }
        
//        createG(g, dispfun_disp, params, n);
	FF(0,n) = cFactor*vel0*sin(omega*t);
        FFsolve(n) = FF(n);
	for (int dof=0; dof<5; dof++)
	    Ftotal(dof) = grav_factor*Ff(dof)+FF(dof,n);
	Delta_d = d-d_last.t();
        
        
        //predictors
        dtilde = d + dt * v + pow(dt,2) * (1.0-2.0*beta) * a/2.0;
        vtilde = v + dt * (1.0-gamma) * a;

	RHS = Ftotal-C*vtilde-K*dtilde;
               
        a = solve(G,RHS);
        d = dtilde+beta*dt*dt*a;
        v = vtilde+dt*gamma*a;
            
//        d_el_last.zeros(nel,neldof);

#ifndef USE_MPI
        for(el = 0; el < nel; el++) {
#endif
	    stress_el_last = stress_el.slice(el);
	    strain_el_last = strain_el.slice(el);
            temp = conv_to<vec>::from(LM.col(el));
            for(ii = 0; ii < neldof; ii++) {
                I = temp(ii);
                if(I > 0) {
                    d_el(el,ii) = d(I-1);
                    d_el_last(el,ii) = d_last(I-1);
		    Delta_d_el(el,ii) = Delta_d(I-1);
                }
            }

            if (n%print_int==0) {
                n_print++;
            }

            if (femParams.whichConst == 1) {
#ifndef USE_MPI
                for (ip = 0; ip < numips; ip++) {
#endif
                    el_stress_isv(coords.row(el), d_el.row(el), params, el, ip, D, stress_el, strain_el);
#ifndef USE_MPI
                }
#endif
            } else if (femParams.whichConst == 2) {
#ifndef USE_MPI
                for (ip = 0; ip < numips; ip++) {
#endif
		    cout<<"stress calculation start"<<endl;
                    el_stress_ellip3d(outputDir, coords.row(el), d_el.row(el), d_el_last.row(el), params, 
                                      n_print, -1, -1, el, ip, D, stress_dem_el,  stress_el, strain_el, dt, demParams);
#ifndef USE_MPI
                }
#endif
            } else {
                cout << "ERROR: NO SPECIFICED CONSTITUTIVE MODEL" << endl;
                exit(0);
            }
            
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif

            //does this need to be sent to each node?
            if (femParams.whichConst == 1) {
                //el_kd_g2int(coords.row(el), d_el.row(el), params, el, stiff_el.slice(el));
            } else if (femParams.whichConst == 2) {
                //ellip3D DOESN'T CURRENTLY OUTPUT D11!!!!!!! HAD TO CHANGE WHEN UPDATED FRACTURE CODE
                //el_kd_g2int_ellip3d(outputDir, coords.row(el), d_el.row(el), params, n, el, stiff_el.slice(el));
            } else {
                cout << "ERROR: NO SPECIFICED CONSTITUTIVE MODEL" << endl;
                exit(0);
            }

//            fs_el.col(el) = el_f_g2int(coords.row(el),stress_el.slice(el),params);
            
//            F_S_el = fs_el.col(el);
               
//            K_el = stiff_el.slice(el);
/*
            temp = conv_to<vec>::from(LM.col(el));
            for(ii = 0; ii < neldof; ii++) {
                I = temp(ii);
                if(I > 0) {
                    F_S(I-1) = F_S(I-1) + F_S_el(ii);
                    for(jj = 0; jj < neldof; jj++) {
                        J = temp(jj);
                        if(J > 0) {
                            K(I-1,J-1) = K(I-1,J-1) + K_el(ii,jj);
                        }
                   }
               }
            }
*/
#ifndef USE_MPI
        }
#endif
/*
#ifdef USE_MPI
        for(ii = 0; ii < K.n_rows; ii++) {
           	for(jj = 0; jj < K.n_cols; jj++) {
           		 MPI_Allreduce(&K(ii,jj),&K_total,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
           		 K(ii,jj) = K_total;
           	}
        }

        for(ii = 0; ii < F_S.n_elem; ii++) {
           	MPI_Allreduce(&F_S(ii), &F_S_total, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
         	F_S(ii) = F_S_total;
        }
#endif
*/
	  d_last = d.t();
//        a = solve(M+gamma*dt*C,-(F_S-F_F-F_G)-C*v);
//        v = v + dt*gamma*a;

#ifdef USE_MPI
        for(ii = 0; ii < stress_el.n_rows; ii++) {
        	for(jj = 0; jj < stress_el.n_cols; jj++) {
        		for(kk = 0; kk < stress_el.n_slices; kk++) {
        			MPI_Allreduce(&stress_el(ii,jj,kk),&K_total,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        			stress_el(ii,jj,kk) = K_total;
        		}
        	}
        }

        for(ii = 0; ii < strain_el.n_rows; ii++) {
         	for(jj = 0; jj < strain_el.n_cols; jj++) {
         		for(kk = 0; kk < strain_el.n_slices; kk++) {
         			MPI_Allreduce(&strain_el(ii,jj,kk),&K_total,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
         			strain_el(ii,jj,kk) = K_total;
         		}
         	}
        }
#endif

        if(rank==0) {
            cout << "dd = " << d << endl;
            cout << "v = " << v << endl;
            cout << "a = " << a << endl;
            cout << "t = " << t << endl;
            cout << "stress_el = " << stress_el << endl;
            cout << "strain_el = " << strain_el << endl;
            cout << "F_total = " << Ftotal << endl;
        }

#ifdef USE_MPI
         MPI_Barrier(MPI_COMM_WORLD);
#endif
        
        if(rank==0 && n%print_int==0) {
			chdir(cCurrentPath);
    	    d_file.open("d.txt",ofstream::out | ofstream::app);
    	    v_file.open("v.txt",ofstream::out | ofstream::app);
    	    a_file.open("a.txt",ofstream::out | ofstream::app);
    	    t_file.open("t.txt",ofstream::out | ofstream::app);
    	    stress_dem_file.open("stress_dem.txt",ofstream::out | ofstream::app);
	    stress_file.open("stress.txt",ofstream::out | ofstream::app);
    	    strain_file.open("isv_el.txt",ofstream::out | ofstream::app);
    	    F_S_file.open("F_total.txt",ofstream::out | ofstream::app);

            d_file.precision(5);
            d_file.setf(ios::scientific);
            v_file.precision(5);
    	    v_file.setf(ios::scientific);
            a_file.precision(5);
    	    a_file.setf(ios::scientific);
	    stress_dem_file.precision(5);
            stress_dem_file.setf(ios::scientific);
            stress_file.precision(5);
    	    stress_file.setf(ios::scientific);
            strain_file.precision(5);
    	    strain_file.setf(ios::scientific);
            F_S_file.precision(5);
    	    F_S_file.setf(ios::scientific);
	    
            d.raw_print(d_file);
    	    v.raw_print(v_file);
    	    a.raw_print(a_file);
            Ftotal.raw_print(F_S_file);
    	    t_file << t << endl;
			for (ii=0; ii<nel; ii++){
				for (jj=0; jj<nstress; jj++){
					for (kk=0; kk<numips; kk++){
						stress_file << stress_el(kk,jj,ii) << " ";
					}
					for (kk=0; kk<numips; kk++){
                                                stress_dem_file << stress_dem_el(kk,jj,ii) << " ";
                                        }
					for (ll=0; ll<numips; ll++){
						strain_file << strain_el(ll,jj,ii) << " ";
					}
				}
			}
			stress_dem_file << endl;
			stress_file << endl;
			strain_file << endl;

    	    d_file.close();
    	    v_file.close();
    	    a_file.close();
    	    t_file.close();
	    stress_dem_file.close();
     	    stress_file.close();
    	    strain_file.close();
    	    F_S_file.close();
        }

#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        stress_el.zeros(numips,nstress,nel);
        strain_el.zeros(numips,nstress,nel);
    }
    
#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
