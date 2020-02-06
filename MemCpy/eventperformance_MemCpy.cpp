#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>

#include "../lcutil.h"


#define COMP_ITERATIONS (4096) //512
#define REGBLOCK_SIZE (4)
#define UNROLL_ITERATIONS (32)
#define THREADS_WARMUP (1024)
int THREADS;
int BLOCKS;
#define deviceNum (0)


//CODE
__global__ void warmup(int aux){

	__shared__ double shared[THREADS_WARMUP];

	short r0 = 1.0,
		  r1 = r0+(short)(31),
		  r2 = r0+(short)(37),
		  r3 = r0+(short)(41);

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to doubleing point 8 operations (4 multiplies + 4 additions)
			r0 = r0 * r0 + r1;//r0;
			r1 = r1 * r1 + r2;//r1;
			r2 = r2 * r2 + r3;//r2;
			r3 = r3 * r3 + r0;//r3;
		}
	}
	shared[threadIdx.x] = r0;
}

template <class T> __global__ void benchmark(T* cdin,  T* cdout){

	const long ite=blockIdx.x * THREADS + threadIdx.x;

	T r0;

	r0=cdin[ite];
	cdout[ite]=r0;
}

void initializeEvents(hipEvent_t *start, hipEvent_t *stop){
	HIP_SAFE_CALL( hipEventCreate(start) );
	HIP_SAFE_CALL( hipEventCreate(stop) );
	HIP_SAFE_CALL( hipEventRecord(*start, 0) );
}

double finalizeEvents(hipEvent_t start, hipEvent_t stop){
	HIP_SAFE_CALL( hipGetLastError() );
	HIP_SAFE_CALL( hipEventRecord(stop, 0) );
	HIP_SAFE_CALL( hipEventSynchronize(stop) );
	float kernel_time;
	HIP_SAFE_CALL( hipEventElapsedTime(&kernel_time, start, stop) );
	HIP_SAFE_CALL( hipEventDestroy(start) );
	HIP_SAFE_CALL( hipEventDestroy(stop) );
	return kernel_time;
}

void runbench_warmup(){
	const int BLOCK_SIZE = 256;
	const int TOTAL_REDUCED_BLOCKS = 256;
    int aux=0;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	hipLaunchKernelGGL((warmup), dim3(dimReducedGrid), dim3(dimBlock ), 0, 0, aux);
	HIP_SAFE_CALL( hipGetLastError() );
	HIP_SAFE_CALL( hipDeviceSynchronize() );
}

void runbench(double* kernel_time, double* flops, double * hostIn, double * hostOut){

	hipEvent_t start, stop;
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);

	initializeEvents(&start, &stop);

    hipLaunchKernelGGL((benchmark<double>), dim3(dimGrid), dim3(dimBlock ), 0, 0, (double *) hostIn, (double *) hostOut);

	hipDeviceSynchronize();

	double time = finalizeEvents(start, stop);
	
	*kernel_time = time;

}

int main(int argc, char *argv[]){

	int i;
	int device = 0;

	hipDeviceProp_t deviceProp;

	printf("Usage: %s [device_num] [metric_name]\n", argv[0]);
	int ntries;
	unsigned int size, sizeB; 
	if (argc > 2){
		size = atoi(argv[1]);
		sizeB = atoi(argv[1])*sizeof(double);
		ntries = atoi(argv[2]);
	}
	else if(argc > 1) {
		size = atoi(argv[1]);
		sizeB = atoi(argv[1])*sizeof(double);
		ntries = 1;
	}
	else {
		printf("No size given!!!\n");
		exit(1);
	}

	if(size >= 1024) {
		if(size % 1024 != 0) {
			printf("Size not divisible by 1024!!!\n");
			exit(1);
		}
	}

	hipSetDevice(deviceNum);

	double time[ntries][2], value[ntries][4];


	// DRAM Memory Capacity
	size_t freeCUDAMem, totalCUDAMem;
	HIP_SAFE_CALL(hipMemGetInfo(&freeCUDAMem, &totalCUDAMem));
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
	printf("Buffer size: %luMB\n", size*sizeof(double)/(1024*1024));

	HIP_SAFE_CALL(hipGetDeviceProperties(&deviceProp, device));

	// Kernel Config
	if(size < 1024) {
		THREADS = size;
		BLOCKS = 1;
	}
	else {
		THREADS = 1024;
		BLOCKS = size/1024;
	}	

	// Initialize Host Memory
	double *hostIn = (double *) malloc(size * sizeof(double));
	double *hostOut = (double *) calloc(size, sizeof(double));

	// Initialize the input data
	for (i = 0; i < size; i++) {
		hostIn[i] = (double) i*100.0f;
	}

	// Initialize Host Memory
	double* deviceIn;
	double* deviceOut;
	HIP_SAFE_CALL(hipMalloc((void**)&deviceIn, size * sizeof(double)));
	HIP_SAFE_CALL(hipMalloc((void**)&deviceOut, size * sizeof(double)));

	// Transfer data from host to device
	HIP_SAFE_CALL(hipMemcpy(deviceIn, hostIn, size*sizeof(double), hipMemcpyHostToDevice));

	// Synchronize in order to wait for memory operations to finish
	HIP_SAFE_CALL(hipDeviceSynchronize());

	int status = system("ls");

	for (i=0;i<1;i++){
		runbench_warmup();
	}
	for (i=0;i<ntries;i++){
		runbench(&time[0][0],&value[0][0], deviceIn, deviceOut);
		printf("Registered time: %f ms\n", time[0][0]);
	}

	// Synchronize in order to wait for memory operations to finish
	HIP_SAFE_CALL(hipDeviceSynchronize());

	// Transfer data from device to host
	HIP_SAFE_CALL(hipMemcpy(hostOut, deviceOut, size*sizeof(double), hipMemcpyDeviceToHost));

	// Synchronize in order to wait for memory operations to finish
	HIP_SAFE_CALL(hipDeviceSynchronize());

	HIP_SAFE_CALL( hipDeviceReset());
	return 0;
}
