#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void histo_kernel ( unsigned char *buffer, long size, int *histo )
{
    __shared__ int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    while (i < size)
    {
        atomicAdd( &temp[buffer[i]], 1);
        i += offset;
    }
    __syncthreads();


    atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}



__constant__ int dev_lut[256];

__global__ void histo_equalization_kernel ( unsigned char *buffer, long size, int *histo, unsigned char *output ) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    while (i < size) {
        if ( dev_lut[buffer[i]] > 255)
            output[i] = 255;
        else
            output[i] = (unsigned char) dev_lut[buffer[i]];

        i += offset;
    }
}



typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    


PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);
void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin);
void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);


int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);
    run_cpu_gray_test(img_ibuf_g, argv[2]);
    free_pgm(img_ibuf_g);
    cudaDeviceReset();

	return 0;
}



void run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    struct timespec  tv1, tv2, tv4, tv5, tv6;
    
    printf("------------------------------------------------------------\n");
    printf("Starting CPU processing...\n");
    printf("------------------------------------------------------------\n");
    

    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);

    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    printf ("Time to generate Hist in CPU: \t\t%g s\n",
            (double) (tv4.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
           (double) (tv4.tv_sec - tv1.tv_sec));

    printf ("Time for Hist Equalization in CPU: \t%g s\n",
            (double) (tv2.tv_nsec - tv4.tv_nsec) / 1000000000.0 +
           (double) (tv2.tv_sec - tv4.tv_sec));

    printf ("Time for the CPU: \t\t\t%g s\n",
            (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
           (double) (tv2.tv_sec - tv1.tv_sec));

    
    printf("------------------------------------------------------------\n");
    printf("Starting GPU processing...\n");
    printf("------------------------------------------------------------\n");

    PGM_IMG result2;
    int histo[256];
    
    result2.w = img_in.w;
    result2.h = img_in.h;
    result2.img = (unsigned char *)malloc(result2.w * result2.h * sizeof(unsigned char));

    cudaEvent_t start, stop;
    gpuErrchk( cudaEventCreate( &start ) );
    gpuErrchk( cudaEventCreate( &stop ) );
    gpuErrchk( cudaEventRecord( start, 0 ) );


    unsigned char *dev_buffer, *dev_output;
    int *dev_histo;
    gpuErrchk( cudaMalloc( (void**)&dev_buffer, img_in.w * img_in.h * sizeof(unsigned char) ) );
    gpuErrchk( cudaMemcpy( dev_buffer, img_in.img, img_in.w * img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMalloc( (void**)&dev_histo,256 * sizeof( int ) ) );
    gpuErrchk( cudaMemset( dev_histo, 0,256 * sizeof( int ) ) );

    cudaDeviceProp prop;
    gpuErrchk( cudaGetDeviceProperties( &prop, 0 ) );
    int blocks = prop.multiProcessorCount;
    printf("Blocks = %d\n", blocks);

    dim3 grid ((img_in.w)/8, (img_in.h)/8);
    dim3 block (8, 8);

    histo_kernel <<<blocks*256, 256>>>( dev_buffer, img_in.w * img_in.h, dev_histo );
    //histo_kernel <<<(img_in.w * img_in.h)/(256*8), 256>>>( dev_buffer, img_in.w * img_in.h, dev_histo );
    //histo_kernel_2 <<<grid, block>>>( dev_buffer, img_in.w, img_in.h , dev_histo );

    gpuErrchk ( cudaMemcpy( histo, dev_histo, 256 * sizeof( int ), cudaMemcpyDeviceToHost ) );


    gpuErrchk ( cudaEventRecord( stop, 0 ) );
    gpuErrchk ( cudaEventSynchronize( stop ) );
    float elapsedTime;
    gpuErrchk ( cudaEventElapsedTime( &elapsedTime,start, stop ) );
    printf( "Time to generate Histo in GPU: \t\t%f s\n", elapsedTime / 1000 );

    
    // verify that we have the same counts via CPU
    long int histoCount = 0;
    for (int i=0; i<256; i++) {
        histoCount += hist[i];
    }
    printf( "Histogram Sum in CPU: %ld\n", histoCount );

    histoCount = 0;
    for (int i=0; i<256; i++) {
        histoCount += histo[i];
    }
    printf( "Histogram Sum in GPU: %ld\n", histoCount );


    gpuErrchk ( cudaEventDestroy( start ) );
    gpuErrchk ( cudaEventDestroy( stop ) );
    gpuErrchk( cudaEventCreate( &start ) );
    gpuErrchk( cudaEventCreate( &stop ) );

    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv5);

    int *lut = (int *)malloc(sizeof(int)*256);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = histo[i++];
    }
    d = img_in.w * img_in.h - min;
    #pragma unroll
    for(i = 0; i < 256; i++){
        cdf += histo[i];
        //lut[i] = (cdf - min)*(256 - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }    
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv6);
    printf ("\t\t\t\t\t(%g) s\n",
            (double) (tv6.tv_nsec - tv5.tv_nsec) / 1000000000.0 +
           (double) (tv6.tv_sec - tv5.tv_sec));


    gpuErrchk( cudaEventRecord( start, 0 ) );

    gpuErrchk( cudaMalloc( (void**)&dev_output, img_in.w * img_in.h * sizeof(unsigned char) ) );
    gpuErrchk( cudaMemcpyToSymbol(dev_lut, lut, 256 * sizeof(int)) );


    histo_equalization_kernel <<<blocks*512, 512>>>( dev_buffer, img_in.w * img_in.h, dev_histo, dev_output );

    gpuErrchk ( cudaMemcpy( result2.img, dev_output, img_in.w * img_in.h * sizeof( unsigned char ), cudaMemcpyDeviceToHost ) );
    

    gpuErrchk ( cudaEventRecord( stop, 0 ) );
    gpuErrchk ( cudaEventSynchronize( stop ) );
    float elapsedTime2;
    gpuErrchk ( cudaEventElapsedTime( &elapsedTime2,start, stop ) );
    printf( "Time for Hist Equalization in GPU: \t%f s\n", elapsedTime2 / 1000 );
    printf( "Time for GPU: \t\t\t\t%f s\n", (elapsedTime + elapsedTime2) / 1000 + ((double) (tv6.tv_nsec - tv5.tv_nsec) / 1000000000.0 +
           (double) (tv6.tv_sec - tv5.tv_sec)) );


    int ret = memcmp(result.img, result2.img, img_in.w * img_in.h);
    printf("Compare Histo Equal: \t%d\n", ret);

    printf("------------------------------------------------------------\n");
    printf("GPU is %f faster than CPU\n", ((double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
           (double) (tv2.tv_sec - tv1.tv_sec))/((elapsedTime + elapsedTime2) / 1000));
    printf("------------------------------------------------------------\n");

    write_pgm(result2, out_filename);
    free_pgm(result2);
    gpuErrchk ( cudaEventDestroy( start ) );
    gpuErrchk ( cudaEventDestroy( stop ) );
    gpuErrchk ( cudaFree( dev_histo ) ) ;
    gpuErrchk ( cudaFree( dev_buffer ) ) ;
    gpuErrchk ( cudaFree( dev_output ) ) ;
    free(lut);
}


PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}





void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }    
    }
    
    /* Get the result image */
    for(i = 0; i < img_size; i++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }       
    }
}