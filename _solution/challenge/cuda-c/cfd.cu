#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <queue>
#include "arraymalloc.h"
#include "boundary.h"
#include "jacobi.h"
#include "cfdio.h"

int main(int argc, char **argv)
{
  int printfreq=1000; //output frequency
  double *error, *bnorm;
  double tolerance=0.0; //tolerance for convergence. <=0 means do not check

  //main arrays
  double *psi, *zet;
  //temporary versions of main arrays
  double *psitmp, *zettmp;

  //command line arguments
  int scalefactor, numiter;

  double re; // Reynold's number - must be less than 3.7

  //simulation sizes
  int bbase=10;
  int hbase=15;
  int wbase=5;
  int mbase=32;
  int nbase=32;

  int irrotational = 1, checkerr = 0;

  int m,n,b,h,w;
  int iter;
  int i,j;

  double tstart, tstop, ttot, titer;

  //do we stop because of tolerance?
  if (tolerance > 0) {checkerr=1;}

  //check command line parameters and parse them

  if (argc <3|| argc >4)
    {
      printf("Usage: cfd <scale> <numiter> [reynolds]\n");
      return 0;
    }

  scalefactor=atoi(argv[1]);
  numiter=atoi(argv[2]);

  if (argc == 4)
    {
      re=atof(argv[3]);
      irrotational=0;
    }
  else
    {
      re=-1.0;
    }

  if(!checkerr)
    {
      printf("Scale Factor = %i, iterations = %i\n",scalefactor, numiter);
    }
  else
    {
      printf("Scale Factor = %i, iterations = %i, tolerance= %g\n",scalefactor,numiter,tolerance);
    }

  if (irrotational)
    {
      printf("Irrotational flow\n");
    }
  else
    {
      printf("Reynolds number = %f\n",re);
    }

  //Calculate b, h & w and m & n
  b = bbase*scalefactor;
  h = hbase*scalefactor;
  w = wbase*scalefactor;
  m = mbase*scalefactor;
  n = nbase*scalefactor;

  re = re / (double)scalefactor;

  printf("Running CFD on %d x %d grid in cuda\n",m,n);

  int sizedata = (m+2)*(n+2)*sizeof(double);
  //Allocate memory on GPU
  cudaMalloc((void**)&psi, sizedata);
  cudaMalloc((void**)&psitmp, sizedata);
  cudaMalloc((void**)&error, sizeof(double));
  cudaMalloc((void**)&bnorm, sizeof(double));

    nvtxRangePush("Initialization");
  // NOTE(Qingwen): faster to use memset than loop through array
  cudaMemset(psi, 0, sizedata);
  cudaMemset(psitmp, 0, sizedata);
  cudaMemset(error, 0, sizeof(double));
  cudaMemset(bnorm, 0, sizeof(double));
    nvtxRangePop(); //pop 


  if (!irrotational){
    //allocate arrays
    cudaMalloc((void**)&zet, (sizedata));
    cudaMalloc((void**)&zettmp, (sizedata));

    //zero the zeta array
    nvtxRangePush("Initialization");
    cudaMemset(zet, 0, sizedata);
    nvtxRangePop(); //pop for REading file
  }
  
  //set the psi boundary conditions
    nvtxRangePush("Boundary_PSI");
  boundarypsi<<<dim3(1),dim3(1)>>>(psi,m,n,b,h,w);
  boundarypsi<<<dim3(1),dim3(1)>>>(psitmp,m,n,b,h,w);
    nvtxRangePop(); //pop 
  // Note(Qingwen): no need if all on the same device.
  // cudaDeviceSynchronize(); 


  //compute normalisation factor for error
  dim3 nthreads(16, 16, 1);
  dim3 nblock(( m + nthreads.x -1 ) / nthreads.x, 
              ( n + nthreads.y -1 ) / nthreads.y, 1);
              
    nvtxRangePush("Compute_Normalization");
  initialnorm<<<dim3((m + 1 + nthreads.x) / nthreads.x, (n + 1 + nthreads.y) / nthreads.y, 1),nthreads>>>(psi,bnorm,m,n);
    nvtxRangePop();

  if (!irrotational)
    {
      //update zeta BCs that depend on psi
      boundaryzet<<<dim3(1),dim3(1)>>>(zet,psi,m,n);

      //update normalisation
      nvtxRangePush("Compute_Normalization");
    initialnorm<<<dim3((m + 1 + nthreads.x) / nthreads.x, (n + 1 + nthreads.y) / nthreads.y, 1),nthreads>>>(zet,bnorm,m,n);
      nvtxRangePop();
    }
  
  // no need for now I added them into the func:  *error = sqrt(*error) / sqrt(*bnorm);
  // bnorm=sqrt(bnorm); 

  printf("\nStarting main loop...\n\n");
  
  tstart=gettime();
  nvtxRangePush("Overall_Iteration");

  for(iter=1;iter<=numiter;iter++){

    //calculate psi for next iteration
    nvtxRangePush("JacobiStep");
    if (irrotational)
      jacobistep<<<nblock, nthreads>>>(psitmp,psi,m,n);
    else
      jacobistepvort<<<nblock, nthreads>>>(zettmp,psitmp,zet,psi,m,n,re);
    nvtxRangePop(); //pop 

    //calculate current error if required
    nvtxRangePush("Calculate_Error");
    if (checkerr || iter == numiter){
      deltasq<<<nblock, nthreads>>>(psitmp,psi,m,n, error);
      computerr<<<1, 1>>>(error, bnorm);
    }
    nvtxRangePop();

    //quit early if we have reached required tolerance
    if (checkerr){
      // FIXME: should have better way, but since we didn't run this part no affects on speed now.
      double host_error;
      cudaMemcpy(&host_error, error, sizeof(double), cudaMemcpyDeviceToHost);
      if (host_error < tolerance){
        printf("Converged on iteration %d\n",iter);
        break;
      }
    }

    nvtxRangePush("Switch_Array");
    std::swap(psi, psitmp);
    if (!irrotational)
      std::swap(zet, zettmp);
    nvtxRangePop();
    
    //update zeta BCs that depend on psi
    if (!irrotational)
      boundaryzet<<<dim3(1),dim3(1)>>>(zet,psi,m,n);

    //print loop information

    if(iter%printfreq == 0){
      if (!checkerr)
        printf("Completed iteration %d\n",iter);
      else
        printf("Completed iteration %d, error = \n",iter);
    }
  }// end of iteration loop

  nvtxRangePop(); //pop for Overall_Iteration

  if (iter > numiter) iter=numiter;

  tstop=gettime();

  ttot=tstop-tstart;
  titer=ttot/(double)iter;

  // Important(Qingwen): we only need to sync at the end here:
  cudaDeviceSynchronize();

  //copy error back to host
  double h_error;
  cudaMemcpy(&h_error, error, sizeof(double), cudaMemcpyDeviceToHost);
  double* h_psi;
  cudaHostAlloc((void**)&h_psi, sizedata, cudaHostAllocDefault);
  cudaMemcpy(h_psi, psi, sizedata, cudaMemcpyDeviceToHost);

  printf("\n... finished\n");
  printf("After %d iterations, the error is %g\n",iter,h_error);
  printf("Time for %d iterations was %g seconds\n",iter,ttot);
  printf("Each iteration took %g seconds\n",titer);

  //output results
  writedatafiles(h_psi,m,n, scalefactor);
  writeplotfile(m,n,scalefactor);

  //free un-needed arrays
  cudaFree(psi);
  cudaFree(psitmp);
  cudaFree(error);
  cudaFree(bnorm);
  cudaFreeHost(h_psi);
  
  if (!irrotational){
    cudaFree(zet);
    cudaFree(zettmp);
  }

  printf("... finished\n");

  return 0;
}
