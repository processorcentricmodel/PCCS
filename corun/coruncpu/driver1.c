#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <inttypes.h>
#include <omp.h>
#define ERT_FLOP 2 
#define ERT_TRIALS_MIN 1
#define ERT_WORKING_SET_MIN 1
#define GBUNIT (1024 * 1024 * 1024)

#define REP2(S)        S ;        S
#define REP4(S)   REP2(S);   REP2(S)
#define REP8(S)   REP4(S);   REP4(S) 
#define REP16(S)  REP8(S);   REP8(S) 
#define REP32(S)  REP16(S);  REP16(S)
#define REP64(S)  REP32(S);  REP32(S)
#define REP128(S) REP64(S);  REP64(S)
#define REP256(S) REP128(S); REP128(S)
#define REP512(S) REP256(S); REP256(S)

#define KERNEL1(a,b,c)   ((a) = (a)*(b))
#define KERNEL2(a,b,c)   ((a) = (a)*(b) +c)

void initialize(uint64_t nsize,
                double* __restrict__ A,
                double value)
{
  uint64_t i;
  for (i = 0; i < nsize; ++i) {
    A[i] = value;
  }
}

void kernel(uint64_t nsize,
            uint64_t ntrials,
            double* __restrict__ A,
            int* bytes_per_elem,
            int* mem_accesses_per_elem)
{
  *bytes_per_elem        = sizeof(*A);
  *mem_accesses_per_elem = 2;

  double alpha = 0.5;
  uint64_t i, j;
  for (j = 0; j < ntrials; ++j) {
#pragma unroll (8)
  for (i = 0; i < nsize; ++i) {
      double beta = 0.8;
//if ((ERT_FLOP & 2) == 2)       /* add 2 flops */
//      KERNEL2(beta,A[i],alpha);
//      KERNEL2(beta,A[i],alpha);
//      KERNEL2(beta,A[i],alpha);
      KERNEL2(beta,A[i],alpha);
//      KERNEL1(beta,A[i],alpha);
//      KERNEL1(beta,A[i],alpha);

//if ((ERT_FLOP & 4) == 4)       /* add 4 flops */
//      REP2(KERNEL2(beta,A[i],alpha));
//        REP2(KERNEL2(beta,A[i],alpha));
//if ((ERT_FLOP & 8) == 8)       /* add 8 flops */
//        REP4(KERNEL2(beta,A[i],alpha));
//      REP4(KERNEL2(beta,A[i],alpha));
//  if ((ERT_FLOP & 16) == 16)     /* add 16 flops */
//      REP8(KERNEL2(beta,A[i],alpha));
//if ((ERT_FLOP & 32) == 32 )    /* add 32 flops */      
//        REP16(KERNEL2(beta,A[i],alpha));
//if ((ERT_FLOP & 64) == 64)     /* add 64 flops */
//      REP32(KERNEL2(beta,A[i],alpha));
//if ((ERT_FLOP & 128) == 128)   /* add 128 flops */
//      REP64(KERNEL2(beta,A[i],alpha));
//if ((ERT_FLOP & 256) == 256)   /* add 256 flops */
//      REP128(KERNEL2(beta,A[i],alpha));
//if ((ERT_FLOP & 512) == 512)   /* add 512 flops */
//      REP256(KERNEL2(beta,A[i],alpha));
//if ((ERT_FLOP & 1024) == 1024) /* add 1024 flops */
//     REP512(KERNEL2(beta,A[i],alpha));
      A[i] = beta;
    }
    alpha = alpha * (1 - 1e-8);
  }
}
double getTime()
{
		double time;
		time = omp_get_wtime();
		return time;
}

int main(int argc, char *argv[]) {

		int rank = 0;
		int nprocs = 1;
		int nthreads = 1;
		int id = 0;

		uint64_t TSIZE = 1<<30;
		uint64_t PSIZE = TSIZE / nprocs;

		double * buf = (double *)malloc(PSIZE);

		if (buf == NULL) {
				fprintf(stderr, "Out of memory!\n");
				return -1;
		}
#pragma omp parallel private(id) num_threads(7)
//#pragma omp parallel private(id) 
		{
				id = omp_get_thread_num();
				nthreads = omp_get_num_threads();

				uint64_t nsize = PSIZE / nthreads;
				nsize = nsize & (~(64-1));
				nsize = nsize / sizeof(double);
				uint64_t nid =  nsize * id;

				// initialize small chunck of buffer within each thread
				initialize(nsize, &buf[nid], 1.0);


				double startTime, endTime;
				uint64_t n,nNew;
				uint64_t t;
				int bytes_per_elem;
				int mem_accesses_per_elem;

				n = 1<<22;
				while (n <= nsize) { // working set - nsize
						uint64_t ntrials = nsize / n;
						if (ntrials < 1)
								ntrials = 1;

						for (t = 1; t <= 600; t = t + 1) { // working set - ntrials
				#pragma omp barrier

								if ((id == 0) && (rank==0)) {
										startTime = getTime();
								}
								// C-code
								kernel(n, t, &buf[nid], &bytes_per_elem, &mem_accesses_per_elem);

				#pragma omp barrier

								if ((id == 0) && (rank == 0)) {
										endTime = getTime();
										double seconds = (double)(endTime - startTime);
										uint64_t working_set_size = n * nthreads * nprocs;
										uint64_t total_bytes = t * working_set_size * bytes_per_elem * mem_accesses_per_elem;
										uint64_t total_flops = t * working_set_size * ERT_FLOP;
										// nsize; trials; microseconds; bytes; single thread bandwidth; total bandwidth
										printf("%12" PRIu64 " %12" PRIu64 " %15.3lf %12" PRIu64 " %12" PRIu64 "\n",
										       working_set_size * bytes_per_elem,
										       t,
										       seconds,
										       total_bytes,
										       total_flops);
                                        printf("BW: %15.3lf\n",total_bytes*1.0/seconds/1024/1024/1024);
								} // print
						} // working set - ntrials

						nNew = 2 * n;
						if (nNew == n) {
								nNew = n+1;
						}

						n = nNew;
				} // working set - nsize

		} // parallel region

		free(buf);


		printf("\n");
		printf("META_DATA\n");
		printf("FLOPS          %d\n", ERT_FLOP);

		printf("OPENMP_THREADS %d\n", nthreads);


		return 0;
}
