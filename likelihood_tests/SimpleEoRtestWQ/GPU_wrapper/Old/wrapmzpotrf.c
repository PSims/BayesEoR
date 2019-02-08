
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"     // if you need CUBLAS, include before magma.h
#include "magma.h"
#include "magma_lapack.h"  // if you need BLAS & LAPACK
//#include "zfill.h"         // code to fill matrix; replace with your application code
#include "complex.h"         // A1 complex


// ------------------------------------------------------------
// Solve A * X = B, where A and X are stored in CPU host memory.
// Internally, MAGMA transfers data to the GPU device
// and uses a hybrid CPU + GPU algorithm.
void cpu_interface( int n, int nrhs , magmaDoubleComplex *A1, magmaDoubleComplex *B1, int Print)
//void cpu_interface( int n, int nrhs , double complex *A1, double complex *B1, int Print)
{
	magma_init();
	magma_int_t lda  = n;
	magma_int_t ldb  = lda;
	magma_int_t info = 0;
	char UL='L';

//	magmaDoubleComplex A1 = &A1;
//	magmaDoubleComplex B1 = &B1;

//	magma_zprint(n, n, A1, lda);
//	magma_zprint(n, nrhs, B1, lda);





//	printf( "Print value %d\n",Print );

	if (Print != 0){
		printf( "using MAGMA CPU interface\n" );
	}

	/* ====================================================================
	Performs operation using MAGMA
	=================================================================== */

	if (Print != 0){
	printf( "about to run magma_zpotrf...\n" );
	}

	magma_zpotrf(magma_uplo_const('L'), n, A1, lda, &info );
//	magma_zposv(magma_uplo_const('L'), n, nrhs, A1, lda, B1, ldb, &info );

	if (Print != 0){
		printf( "run magma_zpotrf...\n" );
	}

	if (info != 0){
		printf("magma_zpotrf returned error %d: %s.\n", (int) info, magma_strerror( info ));
	}



	info = 0;
	if (Print != 0){
		printf( "about to run zpotrs...\n" );
	}

//	double complex A1 = &A1;
//	double complex B1 = &B1;

	lapackf77_zpotrs(&UL, &n, &nrhs, A1, &lda, B1, &ldb, &info );

	if (Print != 0){
	printf( "run zpotrs...\n" );
	}

	if (info != 0){
		printf("zpotrs returned error %d: %s.\n", (int) info, magma_strerror( info ));
	}

//	magma_zprint(n, nrhs, B1, lda);

	magma_finalize();



}



// ------------------------------------------------------------

