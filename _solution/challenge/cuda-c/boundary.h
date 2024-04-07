// added by myself
__global__ void computerr(double *error, double *bnorm);
__global__ void initialnorm(double *psi, double *bnorm,int m, int n);

__global__ void boundarypsi(double *psi, int m, int n, int b, int h, int w);
__global__ void boundaryzet(double *zet, double *psi, int m, int n);
