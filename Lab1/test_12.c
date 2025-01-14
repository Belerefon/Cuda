// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE	4096
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"

/* The horizontal and vertical operators to be used in the sobel filter */
char horiz_operator[3][3] = {{-1, 0, 1}, 
                             {-2, 0, 2}, 
                             {-1, 0, 1}};
char vert_operator[3][3] = {{1, 2, 1}, 
                            {0, 0, 0}, 
                            {-1, -2, -1}};
double lookupTable[2000000];

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);

/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row. 			*/
unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];


/* Implement a 2D convolution of the matrix with the operator */
/* posy and posx correspond to the vertical and horizontal disposition of the *
 * pixel we process in the original image, input is the input image and       *
 * operator the operator we apply (horizontal or vertical). The function ret. *
 * value is the convolution of the operator with the neighboring pixels of the*
 * pixel we process.														  */

 
/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.									 */
double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0, t;
	register int i, j;
	unsigned int p;
	int res, res1, res2, temp;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;
	int t1, t2, t3, t4, t6, t7, t8, t9;

	/* The first and last row of the output array, as well as the first  *
     * and last element of each column are not going to be filled by the *
     * algorithm, therefore make sure to initialize them with 0s.		 */
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
	}

	/* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.						  */
	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}
  
	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}  
  
	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}    

	fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
	fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
	fclose(f_in);
	fclose(f_golden);
  
	/* This is the main computation. Get the starting time. */
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
	/* For each pixel of the output image */
	for  (i=1; i<SIZE-1; i+=1 ) {
		temp = i*SIZE;
		t1 = (i + -1)*SIZE - 1;
		t2 = (i + 0)*SIZE - 1;
		t3 = (i + 1)*SIZE - 1;
		t4 = (i + -1)*SIZE;
		t6 = (i + 1)*SIZE;
		t7 = (i + -1)*SIZE + 1;
		t8 = (i + 0)*SIZE + 1;
		t9 = (i + 1)*SIZE + 1;
		for (j=1; j<SIZE-1; j+=1) {
			/* Apply the sobel filter and calculate the magnitude *
			 * of the derivative.								  */
			res1 = 0;
			res1 += input[t1 + j] * horiz_operator[0][0];
			res1 += input[t2 + j] * horiz_operator[1][0];
			res1 += input[t3 + j] * horiz_operator[2][0];
			res1 += input[t7 + j] * horiz_operator[0][2];
			res1 += input[t8 + j] * horiz_operator[1][2];
			res1 += input[t9 + j] * horiz_operator[2][2];
			res2 = 0;
			res2 += input[t1 + j] * vert_operator[0][0];
			res2 += input[t3 + j] * vert_operator[2][0];
			res2 += input[t4 + j] * vert_operator[0][1];
			res2 += input[t6 + j] * vert_operator[2][1];
			res2 += input[t7 + j] * vert_operator[0][2];
			res2 += input[t9 + j] * vert_operator[2][2];
			p = (res1 * res1) + (res2 * res2) ;	
			res = (int)lookupTable[p];
			output[temp + j] = (unsigned char)res;
		}
	}

	/* Now run through the output and the golden output to calculate *
	 * the MSE and then the PSNR.									 */
	for (i=1; i<SIZE-1; i++) {
		temp = i*SIZE;
		for ( j=1; j<SIZE-1; j+=7 ) {
			t = (output[temp+j] - golden[temp+j]) * (output[temp+j] - golden[temp+j]) ;
			PSNR += t;
			t = (output[temp+j+1] - golden[temp+j+1]) * (output[temp+j+1] - golden[temp+j+1]) ;
			PSNR += t;
			t = (output[temp+j+2] - golden[temp+j+2]) * (output[temp+j+2] - golden[temp+j+2]) ;
			PSNR += t;
			t = (output[temp+j+3] - golden[temp+j+3]) * (output[temp+j+3] - golden[temp+j+3]) ;
			PSNR += t;
			t = (output[temp+j+4] - golden[temp+j+4]) * (output[temp+j+4] - golden[temp+j+4]) ;
			PSNR += t;
			t = (output[temp+j+5] - golden[temp+j+5]) * (output[temp+j+5] - golden[temp+j+5]) ;
			PSNR += t;
			t = (output[temp+j+6] - golden[temp+j+6]) * (output[temp+j+6] - golden[temp+j+6]) ;
			PSNR += t;
		}
	}
  
	PSNR /= (double)(SIZE*SIZE);
	PSNR = 10*log10(65536/PSNR);

	/* This is the end of the main computation. Take the end time,  *
	 * calculate the duration of the computation and report it. 	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

  
	/* Write the output file */
	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
	fclose(f_out);
  
	return PSNR;
}


int main(int argc, char* argv[])
{	
	int i;
	for (i = 0 ; i < 65026 ; i++) {
		lookupTable[i] = sqrt(i);
	}
	for (i = 65026 ; i < 2000000 ; i++) {
		lookupTable[i] = 255;
	}
	double PSNR;
	PSNR = sobel(input, output, golden);
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}

