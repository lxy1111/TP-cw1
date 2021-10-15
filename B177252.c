#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> 

# define NPOINTS 2000
# define MAXITER 400



int main(){

  double a[NPOINTS][NPOINTS], b[NPOINTS][NPOINTS], x[NPOINTS][NPOINTS];
  double diff,  tmp; 
  double start, finish; 

  start = omp_get_wtime();
  
#pragma omp parallel default(none) shared(x,a,b)
  { 
  #pragma omp for nowait
  for (int i=0; i<NPOINTS; i++) {
    for (int j=0; j<NPOINTS; j++) {
       a[i][j] = 1.0; 
       b[i][j] = 3.14159265359; 
       x[i][j] = 1.0; 
    }
  }
  }
  
  
  for (int iters = 0; iters < MAXITER; iters++){
    for (int i=1; i<NPOINTS; i++) {
 #pragma omp parallel  default(none) shared(x,b,a,i) private(tmp)
      {
      #pragma omp for nowait
      for (int j=0; j<NPOINTS; j++) {
  	tmp = a[i][j]/b[i-1][j];
        x[i][j] -= x[i-1][j] * tmp;
	b[i][j] -= a[i][j] * tmp; 
      }
      }
    }

    diff = 0.0;
    
#pragma omp parallel  default(none) shared(x,b,a) reduction(+:diff) private(tmp)
    {
     #pragma omp for nowait
    for (int i=0; i<NPOINTS; i++) {
      for (int j=1; j<NPOINTS; j++) {
	tmp = a[i][j]/b[i][j-1];
        x[i][j] -= x[i][j-1] * tmp;
	b[i][j] -= a[i][j] * tmp; 
	diff += fabs(a[i][j] * tmp)/(NPOINTS*NPOINTS);
      }
    }
    }

  }

  finish = omp_get_wtime();  

/*
 *  Output the results
 */

  printf("Diff = %12.8f \n",diff);
  printf("Time = %12.8f seconds\n",finish-start);

  }
