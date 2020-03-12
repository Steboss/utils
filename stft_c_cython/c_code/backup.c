#include <stdio.h>
#include <stdlib.h>
#include <fftw.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265

void hamming(int windowLength, double *buffer){
  for(int i=0; i<windowLength; i++){
    buffer[i] = 0.54 - (0.46*cos(2*PI*(i/((windowLength-1)))));
    //printf("%.2f\n", buffer[i]);

  }
}

//the main function should compute the  stft of the signal, return the stft
//and the correlation points between a chosen window --for the moment --
//and all the other windows
double stft(float *wav_data, int wav_length, int windowSize, int hop_size)
{
  //double *wav_data should be a 1D array
  //wav_length is the length of the wave data
  //windowSize is the size of the window  for the stft
  //hop_size is the hopping size
  printf("Initialization of parameters...");
  //double *hamming_result ; //define the hamming window
  //hamming_result = malloc(sizeof(double));

  fftw_complex *stft_data, *fft_result, *ifft_result; //do we need inverse Fourier?
  //fftw_complex stft_data[windowSize], fft_result[windowSize], ifft_result[windowSize];
  fftw_plan plan_forward, plan_backward;
  int i ;

  //stft_data = (fftw_complex*) fftw_malloc(sizeof(double)*windowSize); //INPUT PARAM: windowSize
  //fft_result = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*windowSize);
  stft_data = static_cast<double*      >(fftw_malloc(input_size  * sizeof(double)));
  ifft_result = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*windowSize);
  //create the plan to project th eFFT
  plan_forward = fftw_plan_dft_1d(windowSize, stft_data, fft_result, FFTW_FORWARD, FFTW_ESTIMATE);

  //window to be applied to data
  //double hamming_result
  //printf("Creation of a hamming window...");
  //hamming(windowSize, hamming_result);

  //now here we need to implement the stft
  int chunkPosition = 0;
  int readIndex ;
  int numChunks = 0 ;
  /*
  while (chunkPosition < wav_length ){

    for(i=0; i<windowSize; i++){

      readIndex = chunkPosition + i;
      //printf("Index position %d\n", readIndex);

      if (readIndex < wav_length){
        //stft_data[i][0] = wav_data[readIndex];//*hamming_result[i];
        //stft_data[i][1] = 0.0;
        stft_data[i] = wav_data[readIndex];
      }
      else{
        //if we are beyond the wav_length
        stft_data[i] = 0.0 ;
        //stft_data[i][0] = 0.0 ;
        //stft_data[i][1] = 0.0 ; //padding
        break;
      }
    }
    //here process the window
    fftw_execute(plan_forward);

    //store the stft in a data structure
    for (i=0; i<windowSize/2; i++)
    {
      //printf("%2.2f, %2.2f\n", fft_result[i][0], fft_result[i][1]);
      printf("%2.2f\n", fft_result[i]);

      //stft_data[i] = fft_result[i];//[0];
      //stft_result = fft_result[i][1];
    }


    chunkPosition += hop_size;
    numChunks++ ;
    printf("Fourier transform done");
  }

  */
  //sanity check
  /*printf("These are the wav_data\n");
  for (i=0; i< windowSize; i++)
  {
    printf("%.2f\n", wav_data[i]);
  }
  exit(0);*/
  stft_data[0][0] = (*wav_data)[0]  ;
  stft_data[1][0] = 0.0 ;

  //sanity check
  printf("This is the translation into a complex object");
  for ( i=0; i<windowSize;i++)
  {
    printf("%.2f\n", stft_data[i]);
  }
  exit(0);
  //perform the fourier transform
  printf("Fourier transform call...\n");
  fftw_execute(plan_forward);
  printf("Fourier transform done\n");
  for (i=0; i<windowSize/2; i++)
  {
    printf("%.2f\n", fft_result[i]);
  }
  //clean up the memory
  /*
  fftw_destroy_plan(plan_forward);
  fftw_free(stft_data);
  fftw_free(fft_result);
  fftw_free(ifft_result);//do we need a ifft?
  //free(hamming_result);*/


}
