#include <stdio.h>

// Qingwen:
#include <algorithm>
#include <vector>
#include <atomic>
#include <execution>
#include <thrust/iterator/counting_iterator.h>

#include "jacobi.h"

void jacobistep(double *psinew, double *psi, int m, int n)
{
  
	std::for_each(std::execution::par, 
        thrust::counting_iterator<unsigned int>(0u), 
        thrust::counting_iterator<unsigned int>(m * n),
        [m, n, psinew, psi](unsigned int index) {
    int i = index / n + 1;
    int j = index % n + 1;
    psinew[i * (m + 2) + j] = 0.25 * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] + psi[i * (m + 2) + j - 1] + psi[i * (m + 2) + j + 1]);
  });

  // std::vector<unsigned int> indices(m * n);
	// std::generate(indices.begin(), indices.end(), [n = 0]() mutable { return n++; });
  // std::for_each(std::execution::par, indices.begin(), indices.end(),
  //           [m, n, psinew, psi](unsigned int index) {
  //             int i = index / n + 1;
  //             int j = index % n + 1;
  //             psinew[i * (m + 2) + j] = 0.25 * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] + psi[i * (m + 2) + j - 1] + psi[i * (m + 2) + j + 1]);
  //           });
  // int i, j;
  // for(i=1;i<=m;i++)
  //   {
  //     for(j=1;j<=n;j++)
	// {
	//   psinew[i*(m+2)+j]=0.25*(psi[(i-1)*(m+2)+j]+psi[(i+1)*(m+2)+j]+psi[i*(m+2)+j-1]+psi[i*(m+2)+j+1]);
  //       }
  //   }
}

void jacobistepvort(double *zetnew, double *psinew,
		    double *zet, double *psi,
		    int m, int n, double re)
{
  int i, j;

  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  psinew[i*(m+2)+j]=0.25*(  psi[(i-1)*(m+2)+j]+psi[(i+1)*(m+2)+j]+psi[i*(m+2)+j-1]+psi[i*(m+2)+j+1]
			     - zet[i*(m+2)+j] );
	}
    }

  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  zetnew[i*(m+2)+j]=0.25*(zet[(i-1)*(m+2)+j]+zet[(i+1)*(m+2)+j]+zet[i*(m+2)+j-1]+zet[i*(m+2)+j+1])
	    - re/16.0*(
		       (  psi[i*(m+2)+j+1]-psi[i*(m+2)+j-1])*(zet[(i+1)*(m+2)+j]-zet[(i-1)*(m+2)+j])
		       - (psi[(i+1)*(m+2)+j]-psi[(i-1)*(m+2)+j])*(zet[i*(m+2)+j+1]-zet[i*(m+2)+j-1])
		       );
	}
    }
}

double deltasq(double *newarr, double *oldarr, int m, int n)
{
  int i, j;

  double dsq=0.0;
  double tmp;

  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  tmp = newarr[i*(m+2)+j]-oldarr[i*(m+2)+j];
	  dsq += tmp*tmp;
        }
    }

  return dsq;
}
