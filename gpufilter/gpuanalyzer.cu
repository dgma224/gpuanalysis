/* David Mathews
Quick and Dirty Convolution on GPU with FFT
This code has a built in Trapezoidal filter, or can take provided filters in a file


//param file setup
For a trapezoidal filter:
trapfilter np top rise tau

For a custom filter:
custom filelocation np
*/

//general inclusions
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <vector>
#include <algorithm>//included for memory transitions from short to float
#include <complex.h>
#include <chrono>

//CUDA inclusions
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

//defining structures and various types
typedef float2 Complex;

//Incoming waveforms structure
struct WAVEFORM {
  bool result;
  int eventid;
  int board;
  int channel;
  unsigned long timestamp;
  unsigned long req;
  int length;
};
//Parameter File Inputs for Filters
struct PARAMS {
  int batchsize;
  std::string filtertype;
  int top;
  int rise;
  float tau;
  float scale;
  int np;
  int newlen;
  int halflen;
  std::string customloc;
  std::vector<std::string> datafiles;
};
//Function Definitions
void trapfiltergen(float*,PARAMS);//generate trap filter
void filparams(PARAMS&);//get parameters from file
void paramdefault(PARAMS&);//default parameter values in case of file error
void filtergen(float*,PARAMS);
void getcustomfil(float*,PARAMS);
void reverseArray(Complex *array,int start,int end);
int PadFilter(const Complex *, Complex **, int);
void PadData(const short *, short **,int);
double OpenDataFile(WAVEFORM **,short ***,const char *,PARAMS,int&);
void DataOutput(WAVEFORM*,short**, PARAMS, int,int,double);
void getdatafiles(PARAMS&);
void baselineshift(short*,PARAMS);
void waveshift(Complex *, Complex *,PARAMS);
void ShortToComplex(short **, Complex *, int, int, PARAMS);

//debug printing functions
void ComplexDebug(Complex*,int,char*);
void ShortDebug(short*,int,char*);
void FloatDebug(float*,int,char*);

//gpu functions
void GPUConvolution(short**,Complex*,cufftHandle,int,PARAMS,size_t,int);
void CPUGPUOverlapConvolution(short**,Complex*,cufftHandle,int, PARAMS, size_t,int);
void CPUGPUOverlapConvolutionFloat(short**,Complex*,int, PARAMS);
void CPUGPUConvolutionFloat(short**,Complex*,cufftHandle,int,PARAMS,size_t,int);
__global__ void multiplication(Complex*,const Complex *, int, float);

int main(int argc, char* argv[]){
  //no input expected, only takes in information from parameter files
  PARAMS param;
  filparams(param);
  getdatafiles(param);
  int batchsize = param.batchsize;
  //with parameters found now, create the filter
  //create the filter with the larger size, just don't do the shift, it isn't neededn
  param.newlen=param.np;//currently newlen will still be used, just in case shifts need to be applied	
  float *filter = (float*)calloc(param.newlen,sizeof(float));
  filtergen(filter,param); 
  size_t floatone = sizeof(float)*param.newlen;
  size_t floatmul = sizeof(float)*param.newlen*batchsize;
  //setup np/2+1 lengths,CUFFT_R2C decreases by this size
  param.halflen = param.newlen/2+1;
  //size_t onesize = sizeof(Complex)*param.halflen;
  size_t bulksize = sizeof(Complex)*param.halflen*batchsize;
  float *fgpufil,*fgpubulkfil;
  checkCudaErrors(cudaMalloc(&fgpufil,floatone));
  checkCudaErrors(cudaMemcpy(fgpufil,filter,floatone,cudaMemcpyHostToDevice));

  //now copy this filter for as many waveforms as will be used
  checkCudaErrors(cudaMalloc(&fgpubulkfil,floatmul));
  for(int i = 0;i<batchsize;i++){
    checkCudaErrors(cudaMemcpy(&fgpubulkfil[param.newlen*i],fgpufil, floatone, cudaMemcpyDeviceToDevice));
  }
  //now create the FFT plan and execute it on the filter
  //need to first create bulk complex filter storage
  Complex *gpufil;
  checkCudaErrors(cudaMalloc(&gpufil,bulksize));
  cufftHandle plan;
  checkCudaErrors(cufftPlan1d(&plan,param.newlen,CUFFT_R2C,batchsize));
  checkCudaErrors(cufftExecR2C(plan, fgpubulkfil,gpufil));
  //the filter is ready to go for convolution now
  //iterate once per data file
  int numfiles = param.datafiles.size();
  for(int file = 0;file<numfiles;file++){
    int numwaves = 0;
    WAVEFORM * wavedat;
    short **waveforms;
    double startval = OpenDataFile(&wavedat,&waveforms, param.datafiles.at(file).c_str(),param,numwaves);
    //at this point waveforms are ready to be copied over, handle them in batches
    auto start = std::chrono::system_clock::now();
    CPUGPUOverlapConvolutionFloat(waveforms, gpufil, numwaves, param);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> t = end - start;
    std::cout<<param.datafiles.at(file)<<" : "<<numwaves<<" : "<<t.count()<<std::endl;  
    //now output convolved data properly
    DataOutput(wavedat,waveforms,param,file,numwaves,startval);
    free(wavedat);
    for(int i = 0;i<numwaves;i++){
      free(waveforms[i]);
    }
  }
  //free(filter);
  //free(padfilter);
  checkCudaErrors(cudaFree(gpufil));
  checkCudaErrors(cudaFree(fgpubulkfil));
  checkCudaErrors(cudaFree(fgpufil));
  checkCudaErrors(cufftDestroy(plan));
  return 0;
}

void ComplexDebug(Complex *data,int length,char *filename){
  FILE *fout;
  fout = fopen(filename,"w");
  for(int i = 0;i<length;i++){
    fprintf(fout,"%f %f\n",data[i].x,data[i].y);
  }
  fclose(fout);
  std::cout<<filename<<" has been printed"<<std::endl;
}

void ShortDebug(short *data,int length,char *filename){
  FILE *fout;
  fout = fopen(filename,"w");
  for(int i = 0;i<length;i++){
    fprintf(fout,"%d\n",data[i]);
  }
  fclose(fout);
  std::cout<<filename<<" has been printed"<<std::endl;
}

void FloatDebug(float *data,int length,char *filename){
  FILE *fout;
  fout = fopen(filename,"w");
  for(int i = 0;i<length;i++){
    fprintf(fout,"%f\n",data[i]);
  }
  fclose(fout);
  std::cout<<filename<<" has been printed"<<std::endl;
}


void DataOutput(WAVEFORM* wavedat, short **waveforms, PARAMS param, int numfile, int numwaves, double start){
  //this opens a data file for output
  //will take a Run_A_B.bin to a Run_A_B_res.bin
  std::string outname;
  size_t perloc = param.datafiles.at(numfile).find_last_of(".");
  outname = param.datafiles.at(numfile).substr(0,perloc);
  outname = outname + "_res.bin";
  //now open that file and write to it
  FILE * fout;
  fout = fopen(outname.c_str(), "wb");
  if(fout == NULL){
    std::cout<<outname<<" couldn't be opened: Code Exiting"<<std::endl;
    exit(1);
  }
  fwrite(&start,sizeof(double),1,fout);
  for(int i = 0;i<numwaves;i++){
    fwrite(&wavedat[i].result,sizeof(bool),1,fout);
    fwrite(&wavedat[i].eventid,sizeof(int),1,fout);
    fwrite(&wavedat[i].board,sizeof(int),1,fout);
    fwrite(&wavedat[i].channel,sizeof(int),1,fout);
    fwrite(&wavedat[i].req,sizeof(unsigned long),1,fout);
    fwrite(&wavedat[i].timestamp,sizeof(unsigned long),1,fout);
    fwrite(&wavedat[i].length,sizeof(int),1,fout);
    fwrite(waveforms[i],sizeof(short),param.np,fout);
  }
  fclose(fout);  
}

double OpenDataFile(WAVEFORM** wavedat,short*** waveforms,const char *filename,PARAMS param,int &numwaves){
  //first get size of the file and figure out how many waveforms that is
  FILE * fin;
  fin = fopen(filename, "rb");
  if(fin == NULL){
    std::cout<<filename<<" couldn't be opened: Code Exiting"<<std::endl;
    exit(1);
  }
  fseek(fin, 0L, SEEK_END);
	long int size = ftell(fin);
	fseek(fin, 0L, SEEK_SET);
	numwaves = int(((double)size-8)/(7033));
  //now that we know the number of waves, we can allocate our structs and whatnot
  WAVEFORM* tempwavedat=(WAVEFORM*)malloc(sizeof(WAVEFORM)*numwaves);
  short** tempwaves=(short**)malloc((long)numwaves*sizeof(short*));
  for(int i = 0;i<numwaves;i++){
    tempwaves[i]=(short*)calloc(param.np,sizeof(short));
  }
  //now allocate each waveform storage location
  double initval;
  fread(&initval,8,1,fin);
  for(int i = 0;i<numwaves;i++){
    fread(&tempwavedat[i].result,sizeof(bool),1,fin);
    fread(&tempwavedat[i].eventid,sizeof(int),1,fin);
    fread(&tempwavedat[i].board,sizeof(int),1,fin);
    fread(&tempwavedat[i].channel,sizeof(int),1,fin);
    fread(&tempwavedat[i].req,sizeof(unsigned long),1,fin);
    fread(&tempwavedat[i].timestamp,sizeof(unsigned long),1,fin);
    fread(&tempwavedat[i].length,sizeof(int),1,fin);
    fread(tempwaves[i],sizeof(short),param.np,fin);
  }  
  fclose(fin);
  //now the data file has been completely read in to the temporary variable, move it to the final one
  *waveforms=tempwaves;
  *wavedat=tempwavedat;
  return initval;
}

__global__ void multiplication(Complex *data, const Complex *filter, int length, float scale){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < length){
		Complex temp;
		temp.x = (data[i].x * filter[i].x - data[i].y * filter[i].y)*(scale);
		temp.y = (data[i].x * filter[i].y + data[i].y * filter[i].x)*(scale);
		data[i] = temp;
	}
}

void ShortToComplex(short **sdata, Complex *cdata, int startloc, int batchsize, PARAMS param){
  for(int i = 0;i<batchsize;i++){
    for(int j = 0;j<param.np;j++){
      cdata[i*param.newlen+j].x=sdata[i+startloc*batchsize][j];
      cdata[i*param.newlen+j].y=0.0;
    }
    for(int j = param.np;j<param.newlen;j++){
      cdata[i*param.newlen+j].x=0.0;
      cdata[i*param.newlen+j].y=0.0;
    }
  }
}
 
void ShortToFloat(short **sdata, float *fdata, int startloc, int batchsize, PARAMS param){
  for(int i = 0;i<batchsize;i++){
    std::copy(&sdata[i+startloc*batchsize][0],&sdata[i+startloc*batchsize][param.np-1], &fdata[i*param.newlen]);
    //memset(&fdata[i*param.newlen+param.np], 0, (param.newlen-param.np)*sizeof(float));  
  } 
}

void CPUGPUOverlapConvolutionFloat(short** hostwaves, Complex *gpufilter, int numwaves, PARAMS param){
  //this is effectively the same function as the CPUGPUOverlapConvolution
  //but this uses less host ram and may be less CPU intensive due to using
  //float to complex and complex to float fourier transforms instead of
  //complex to complex
  //Everything else is the same
  //in this case, we actually don't use the cufftHandles given in the input parameters
  int batchsize=param.batchsize;
  size_t bulksize = param.batchsize*sizeof(Complex)*param.halflen;
  int numbatch=numwaves/batchsize;
  int leftover=numwaves%batchsize;
  int lastbatch=0;
  if(numbatch%2==1){//if we have an odd number of batches
    numbatch-=1;    
    lastbatch=1;
  }
  int threads = 1024;
  int blocks = 0;
  if((batchsize*param.halflen)%threads==0){
    blocks=batchsize*param.halflen/threads;
  }
  else{
    blocks=batchsize*param.halflen/threads+1;
  }
  //we want two different streams to be running at a time that are overlapping
  //for the first one, is just the very first data set normally
  float scale = param.scale/((float)param.newlen);
  float *tempwave1;
  float *tempwave2;
  size_t floatgpusize = batchsize*param.newlen*sizeof(float);
  checkCudaErrors(cudaMallocHost(&tempwave1,floatgpusize));
  checkCudaErrors(cudaMallocHost(&tempwave2,floatgpusize));

  float *prefft1, *prefft2;//storage before the fft is applied
  checkCudaErrors(cudaMalloc(&prefft1, floatgpusize));
  checkCudaErrors(cudaMalloc(&prefft2, floatgpusize));
  //setup the cufft plans and streams along with data storage
  Complex *gpuwaves1,*gpuwaves2;//storage after fft is applied
  checkCudaErrors(cudaMalloc(&gpuwaves1,bulksize));
  checkCudaErrors(cudaMalloc(&gpuwaves2,bulksize)); 
  //create two cudaStreams
  cudaStream_t stream1,stream2; 
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  //create the plans, 2 for forward, 2 for reverse
  cufftHandle plan1,plan2,plan1i,plan2i;
  checkCudaErrors(cufftPlan1d(&plan1,param.newlen,CUFFT_R2C,batchsize));
  checkCudaErrors(cufftPlan1d(&plan2,param.newlen,CUFFT_R2C,batchsize));
  checkCudaErrors(cufftPlan1d(&plan1i,param.newlen,CUFFT_C2R,batchsize));
  checkCudaErrors(cufftPlan1d(&plan2i,param.newlen,CUFFT_C2R,batchsize));
  cufftSetStream(plan1,stream1);
  cufftSetStream(plan2,stream2);
  cufftSetStream(plan1i,stream1);
  cufftSetStream(plan2i,stream2);
  checkCudaErrors(cudaDeviceSynchronize());
  for(int i = 0;i<numwaves;i++){
    baselineshift(hostwaves[i],param);
  }
  //now have created both streams and plans, should be good to go
  for(int batch=0;batch<numbatch;batch+=2){//step by two each time
    //wave1 prep 
    ShortToFloat(hostwaves,tempwave1,batch,batchsize,param);
    cudaMemcpyAsync(prefft1,tempwave1,floatgpusize,cudaMemcpyHostToDevice,stream1);
    cufftExecR2C(plan1,prefft1,gpuwaves1);
    multiplication<<<blocks,threads,0,stream1>>>(gpuwaves1,gpufilter,batchsize*param.halflen, scale);
    cufftExecC2R(plan1i,gpuwaves1,prefft1);
    ShortToFloat(hostwaves,tempwave2,batch+1,batchsize,param); 
    cudaMemcpyAsync(prefft2,tempwave2,floatgpusize,cudaMemcpyHostToDevice,stream2);
    cudaMemcpyAsync(tempwave1,prefft1,floatgpusize,cudaMemcpyDeviceToHost,stream1);
    cufftExecR2C(plan2,prefft2,gpuwaves2);
    multiplication<<<blocks,threads,0,stream2>>>(gpuwaves2,gpufilter,batchsize*param.halflen, scale);
    cufftExecC2R(plan2i,gpuwaves2,prefft2);    
    cudaMemcpyAsync(tempwave2,prefft2,floatgpusize,cudaMemcpyDeviceToHost,stream2);
    cudaStreamSynchronize(stream1);
    for(int i = 0;i<batchsize;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+batch*batchsize][j]=tempwave1[i*param.newlen+j];
      }
    } 
    cudaStreamSynchronize(stream2);
    for(int i = 0;i<batchsize;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+(batch+1)*batchsize][j]=tempwave2[i*param.newlen+j];
      }
    }
    
  }  
  //now the main batches are done, now do the extra batch if the number of batches was even
  if(lastbatch==1){
    //we have one more batch to do
    int batch=numbatch-1;
    ShortToFloat(hostwaves,tempwave1,batch,batchsize,param);
    cudaMemcpyAsync(prefft1,tempwave1,floatgpusize,cudaMemcpyHostToDevice,stream1);
    cufftExecR2C(plan1,prefft1,gpuwaves1);
    multiplication<<<blocks,threads,0,stream1>>>(gpuwaves1,gpufilter,batchsize*param.halflen, scale);
    cufftExecC2R(plan1,(cufftComplex*)gpuwaves1,prefft1);
    cudaMemcpyAsync(tempwave1,prefft1,floatgpusize,cudaMemcpyDeviceToHost,stream1);
    for(int i = 0;i<batchsize;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+(batch)*batchsize][j]=tempwave1[i*param.newlen+j];
      }
    } 
  }
  //now do the partial batch at the end
  if(leftover!=0){
    ShortToFloat(hostwaves,tempwave1,numbatch,leftover,param);
    for(int i = leftover;i<batchsize;i++){//fill extra slots with blanks
      for(int j = 0;j<param.newlen;j++){
        tempwave1[i*param.newlen+j]=0.0f;
      }
    }
    cudaMemcpyAsync(prefft1, tempwave1, floatgpusize, cudaMemcpyHostToDevice,stream1);
    cufftExecR2C(plan1,prefft1,gpuwaves1);
		multiplication<<<blocks,threads,0,stream1>>>(gpuwaves1, gpufilter ,batchsize*param.halflen, scale);
    cufftExecC2R(plan1,gpuwaves1,prefft1);
    cudaMemcpyAsync(tempwave1,prefft1,floatgpusize,cudaMemcpyDeviceToHost,stream1);
    checkCudaErrors(cudaStreamSynchronize(stream1));
    //change the final location a bit, but everything else is the same easily enough
    for(int i = 0;i<leftover;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+numbatch*batchsize][j]=tempwave1[i*param.newlen+j];
      }
    }  
  }
  cudaFreeHost(&tempwave1);
  cudaFreeHost(&tempwave2);
  cudaFree(prefft1);
  cudaFree(prefft2);
  cudaFree(gpuwaves1);
  cudaFree(gpuwaves2);
  cufftDestroy(plan1);
  cufftDestroy(plan2);
  cufftDestroy(plan1i);
  cufftDestroy(plan2i);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  
}
     
void CPUGPUOverlapConvolution(short** hostwaves, Complex *gpufilter,cufftHandle plan1, int numwaves, PARAMS param, size_t bulksize,int batchsize){
  //NO LONGER WORKS!!! THIS EXPECTS A DIFFERENT FILTER TYPE ON GPU
  //DO NOT USE UNTIL FILTER IS REDEFINED WITH CUFFT_C2C version
  //this version of the convolution overlaps the CPU work of shifting waveforms
  //and loading them into the complex buffer while the GPU works
  //this doesn't use cudaStreams
  //first figure out how many batches to do
  int numbatch=numwaves/batchsize;
  int leftover=numwaves%batchsize;
  int lastbatch=0;
  if(numbatch%2==1){//if we have an odd number of batches
    numbatch-=1;    
    lastbatch=1;
  }
  int threads = 1024;
  int blocks = 0;
  if((batchsize*param.newlen)%threads==0){
    blocks=batchsize*param.newlen/threads;
  }
  else{
    blocks=batchsize*param.newlen/threads+1;
  }
  //we want two different streams to be running at a time that are overlapping
  //for the first one, is just the very first data set normally
  float scale = param.scale/((float)param.newlen);
  Complex *tempwave1;
  Complex *tempwave2;
  checkCudaErrors(cudaMallocHost(&tempwave1,param.newlen*batchsize*sizeof(Complex)));
  checkCudaErrors(cudaMallocHost(&tempwave2,param.newlen*batchsize*sizeof(Complex)));
  //setup the cufft plans and streams along with data storage
  Complex *gpuwaves1,*gpuwaves2;
  checkCudaErrors(cudaMalloc(&gpuwaves1,bulksize));
  checkCudaErrors(cudaMalloc(&gpuwaves2,bulksize)); 
  //create two cudaStreams
  cudaStream_t stream1,stream2; 
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  //create the other plan
  cufftHandle plan2;
  checkCudaErrors(cufftPlan1d(&plan2,param.newlen,CUFFT_C2C,batchsize));
  cufftSetStream(plan1,stream1);
  cufftSetStream(plan2,stream2);
  checkCudaErrors(cudaDeviceSynchronize());
  //now have created both streams and plans, should be good to go
  for(int i = 0;i<numwaves;i++){
    baselineshift(hostwaves[i],param);
  }
  for(int batch=0;batch<numbatch;batch+=2){//step by two each time
    //wave1 prep 
    //do baseline shift on waveforms

    ShortToComplex(hostwaves,tempwave1,batch,batchsize,param);
    cudaMemcpyAsync(gpuwaves1,tempwave1,bulksize,cudaMemcpyHostToDevice,stream1);
    cufftExecC2C(plan1,(cufftComplex*)gpuwaves1,(cufftComplex*)gpuwaves1,CUFFT_FORWARD);
    multiplication<<<blocks,threads,0,stream1>>>(gpuwaves1,gpufilter,batchsize*param.newlen, scale);
    cufftExecC2C(plan1,(cufftComplex*)gpuwaves1,(cufftComplex*)gpuwaves1,CUFFT_INVERSE);
    //now setup the other waves while we are waiting for that to finish
    ShortToComplex(hostwaves,tempwave2,batch+1,batchsize,param);
    cudaStreamSynchronize(stream2);
    cudaMemcpyAsync(gpuwaves2,tempwave2,bulksize,cudaMemcpyHostToDevice,stream2);
    //now we need to start the analysis on wave2 and copy back the other stuff
    cudaMemcpyAsync(tempwave1,gpuwaves1,bulksize,cudaMemcpyDeviceToHost,stream1);
    //start up the analysis on stream2 now
    cufftExecC2C(plan2,(cufftComplex*)gpuwaves2,(cufftComplex*)gpuwaves2,CUFFT_FORWARD);
    multiplication<<<blocks,threads,0,stream2>>>(gpuwaves2,gpufilter,batchsize*param.newlen, scale);
    cufftExecC2C(plan2,(cufftComplex*)gpuwaves2,(cufftComplex*)gpuwaves2,CUFFT_INVERSE);
    //now store the results from stream 1 after synchronization 
    cudaStreamSynchronize(stream1);
    for(int i = 0;i<batchsize;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+batch*batchsize][j]=tempwave1[i*param.newlen+j].x;
      }
    }  
    //now do the storage of stuff from stream 2
    cudaMemcpyAsync(tempwave2,gpuwaves2,bulksize,cudaMemcpyDeviceToHost,stream2);
    cudaStreamSynchronize(stream2);
    for(int i = 0;i<batchsize;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+(batch+1)*batchsize][j]=tempwave2[i*param.newlen+j].x;
      }
    } 
  }  
  //now the main batches are done, now do the extra batch if the number of batches was even
  if(lastbatch==1){
    //we have one more batch to do
    int batch=numbatch-1;
    ShortToComplex(hostwaves,tempwave1,batch,batchsize,param);
    cufftExecC2C(plan1,(cufftComplex*)gpuwaves1,(cufftComplex*)gpuwaves1,CUFFT_FORWARD);
    multiplication<<<blocks,threads,0,stream1>>>(gpuwaves1,gpufilter,batchsize*param.newlen, scale);
    cufftExecC2C(plan1,(cufftComplex*)gpuwaves1,(cufftComplex*)gpuwaves1,CUFFT_INVERSE);
    cudaMemcpyAsync(tempwave1,gpuwaves1,bulksize,cudaMemcpyDeviceToHost,stream1);
    for(int i = 0;i<batchsize;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+(batch)*batchsize][j]=tempwave2[i*param.newlen+j].x;
      }
    } 
  }
  //now do the partial batch at the end
  if(leftover!=0){
    ShortToComplex(hostwaves,tempwave1,numbatch,leftover,param);
    for(int i = leftover;i<batchsize;i++){//fill extra slots with blanks
      for(int j = 0;j<param.newlen;j++){
        tempwave1[i*param.newlen+j].x=0.0f;
        tempwave1[i*param.newlen+j].y=0.0f;
      }
    }
    cudaMemcpyAsync(gpuwaves1, tempwave1, bulksize, cudaMemcpyHostToDevice,stream1);
    cufftExecC2C(plan1, (cufftComplex*)gpuwaves1,(cufftComplex*)gpuwaves1, CUFFT_FORWARD);
		multiplication<<<blocks,threads,0,stream1>>>(gpuwaves1, gpufilter ,batchsize*param.newlen, scale);
    cufftExecC2C(plan1,(cufftComplex*)gpuwaves1,(cufftComplex*)gpuwaves1,CUFFT_INVERSE);
    cudaMemcpyAsync(tempwave1,gpuwaves1,bulksize,cudaMemcpyDeviceToHost,stream1);
    checkCudaErrors(cudaStreamSynchronize(stream1));
    //change the final location a bit, but everything else is the same easily enough
    for(int i = 0;i<leftover;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+numbatch*batchsize][j]=tempwave1[i*param.newlen+j].x;
      }
    }  
  }
}

void GPUConvolution(short** hostwaves, Complex *gpufilter,cufftHandle plan, int numwaves, PARAMS param, size_t bulksize,int batchsize){
  //NO LONGER WORKS!!! THIS EXPECTS A DIFFERENT FILTER TYPE ON GPU
  //DO NOT USE UNTIL FILTER IS REDEFINED WITH CUFFT_C2C version
  int numbatch=numwaves/batchsize;
  int leftover=numwaves%batchsize;
  checkCudaErrors(cudaSetDevice(0));
  Complex *gpuwaves;
  checkCudaErrors(cudaMalloc(&gpuwaves,bulksize));
  //first make this the simplistic version, then will optimize later
  int threads = 1024;
  int blocks = 0;
  if((batchsize*param.newlen)%threads==0){
    blocks=batchsize*param.newlen/threads;
  }
  else{
    blocks=batchsize*param.newlen/threads+1;
  }
  float scale = param.scale/((float)param.newlen);
  Complex *tempwave;
  checkCudaErrors(cudaMallocHost(&tempwave,param.newlen*batchsize*sizeof(Complex)));
  for(int batch=0;batch<numbatch;batch++){
    //copy waveforms over
    std::cout<<batch<<" : "<<numbatch<<std::endl;
    ShortToComplex(hostwaves,tempwave,batch,batchsize,param);
    checkCudaErrors(cudaMemcpy(gpuwaves, tempwave, bulksize, cudaMemcpyHostToDevice));
    //now do the FFT on these waveforms
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)gpuwaves,(cufftComplex*)gpuwaves,CUFFT_FORWARD));
    //now do the dot product on these waves
		multiplication<<<blocks,threads>>>(gpuwaves, gpufilter ,batchsize*param.newlen,scale);
    //dot product is done, apply the bulk inverse FFT
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cufftExecC2C(plan,(cufftComplex*)gpuwaves,(cufftComplex*)gpuwaves,CUFFT_INVERSE));
    //now move back to host
    checkCudaErrors(cudaMemcpy(tempwave,gpuwaves,bulksize,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    for(int i = 0;i<batchsize;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+batch*batchsize][j]=tempwave[i*param.newlen+j].x;
      }
    }  
  }
  //now for what is left
  if(leftover!=0){
    std::cout<<"analyzing leftover waves"<<std::endl;
    ShortToComplex(hostwaves,tempwave,numbatch,leftover,param);
    for(int i = leftover;i<batchsize;i++){//fill extra slots with blanks
      for(int j = 0;j<param.newlen;j++){
        tempwave[i*param.newlen+j].x=0.0f;
        tempwave[i*param.newlen+j].y=0.0f;
      }
    }
    //Everything is the exact same now, just the output is different
    checkCudaErrors(cudaMemcpy(gpuwaves, tempwave, bulksize, cudaMemcpyHostToDevice));
    //now do the FFT on these waveforms
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)gpuwaves,(cufftComplex*)gpuwaves,CUFFT_FORWARD));
    //copy back for debugging
    checkCudaErrors(cudaMemcpy(tempwave,gpuwaves,bulksize,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    //now do the dot product on these waves
		multiplication<<<blocks,threads>>>(gpuwaves, gpufilter ,batchsize*param.newlen,scale);
    //dot product is done, apply the bulk inverse FFT
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cufftExecC2C(plan,(cufftComplex*)gpuwaves,(cufftComplex*)gpuwaves,CUFFT_INVERSE));
    //now move back to host
    checkCudaErrors(cudaMemcpy(tempwave,gpuwaves,bulksize,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    //change the final location a bit, but everything else is the same easily enough
    for(int i = 0;i<leftover;i++){
      for(int j = 0;j<param.np;j++){
        hostwaves[i+numbatch*batchsize][j]=tempwave[i*param.newlen+j].x;
      }
    }  
  }
  checkCudaErrors(cudaFreeHost(tempwave));
}

void waveshift(Complex *wave, Complex *tempwave,PARAMS param){
  for(int j = 0;j<param.newlen;j++){
		int shift = param.newlen - param.np;
		int newloc = shift + j;
		if(newloc < param.newlen){
			wave[newloc] = tempwave[j];
		}
		else{
			newloc = newloc - param.newlen;
			wave[newloc] = tempwave[j];
		}
	}
}
//do the baseline shift and bit shift for all waveforms
void baselineshift(short* wave,PARAMS param){
  for(int i = 0;i<param.np;i++){
    wave[i]=int(wave[i]) & 16383;
    if(wave[i]>=8192){
      wave[i]=wave[i]-16384;
    }
  }
  //now that bit shift is done, calculate average and do baseline shift
  float ave=0.0f;
  int pretrigger = 900;
  for(int i = 0;i<pretrigger;i++){
    ave+=wave[i];
  }
  ave=ave/(float)pretrigger;
  for(int i = 0;i<param.np;i++){
    wave[i]=wave[i]-ave;
  }      
}

void getdatafiles(PARAMS& param){
  //open the data file location manager, expects file locations in dataloc.dat
  std::ifstream datain;
  datain.open("dataloc.dat");
  std::string line;
  if(datain.is_open()){
    for(std::string i; datain>>i;){
      param.datafiles.push_back(i);
    }
  }
  else{
    std::cout<<"Data Input File failed to open"<<std::endl;
    exit(1);
  }  
  datain.close();
}



//function that controls the generation of the filters
void filtergen(float *filter, PARAMS param){
  if(param.filtertype=="trapfilter"){//if using trapezoidal filter, call that function
    trapfiltergen(filter, param);
  }
  else{//otherwise use a custom filter from a file
    getcustomfil(filter, param);
  }
}

void getcustomfil(float* filter, PARAMS param){
  //open file for custom filter
  std::ifstream filin;
  filin.open(param.customloc.c_str());
  std::string line;
  if(filin.is_open()){
    int loc = 0;
    for(int i; filin>>i;){
      filter[loc]=i;
      loc++;
    }
    while(loc<param.np-1){
      filter[loc]=0.0;
      loc++;
    }
  }
  else{
    std::cout<<"Custom Filter File failed to open: Code Exiting"<<std::endl;
  }
  filin.close();
}

//generate the trapezoidal filter based on parameters given
void trapfiltergen(float *filter, PARAMS param){
  int rise = param.rise;
  int top = param.top;
  int tau = param.tau;
  int np = param.np;
	for (int i=0;i<rise;++i){
		filter[i]=tau+i;
		filter[i+rise+top]=rise-tau-(i);
	}
	for (int i=rise;i<rise+top;++i)
	{
		filter[i]=rise;
	}
	//now pad the filter for the appropriate number of points
	for(int i = rise+rise+top;i<np;i++){
		filter[i] = 0;
	}
}


//setup default parameters
void paramdefault(PARAMS& param){
  param.batchsize=2048;
  param.filtertype="";
  param.top=0;
  param.rise=0;
  param.tau=0.0;
  param.customloc=""; 
  param.np=3500;
  param.newlen=0;
  param.scale=0.0f;
}

//fill up the parameters struct
void filparams(PARAMS& param){
  //open parameter file
  //set default values
  paramdefault(param);
  std::ifstream pfin;
  pfin.open("param.dat");
  if(pfin.is_open()){
    pfin >> param.batchsize;
    pfin >> param.filtertype;
    if(param.filtertype=="trapfilter"){//if the user requested trapezoidal filter get those inputs
      pfin >> param.np;
      pfin >> param.top;
      pfin >> param.rise;
      pfin >> param.tau;
      param.scale = 1.0f/((float)param.rise*param.tau);
    }
    else if(param.filtertype=="custom"){//get custom file location
      pfin >> param.customloc;
      pfin >> param.np;
      pfin >> param.scale;
    }
    else{
      std::cout<<"Improper Filter Request: Code Exiting"<<std::endl;
      exit(1);
    }
  }
  else{
    std::cout<<"param.dat failed to open: Code Exiting"<<std::endl;
    exit(1);
  }
  
  std::cout<<"Input Params:"<<std::endl;
  std::cout<<"\t FilterType: "<<param.filtertype<<std::endl;
  std::cout<<"\t Top: "<<param.top<<" Rise: "<<param.rise<<" Tau: "<<param.tau<<std::endl;
  std::cout<<"\t customloc: "<<param.customloc<<" np: "<<param.np<<std::endl;
  pfin.close();
}

int PadFilter(const Complex *filter, Complex **paddedfilter, int filtersize){
	int minRadius = filtersize / 2;
	int maxRadius = filtersize - minRadius;
	int new_size = filtersize + maxRadius;	
	Complex *new_data = (Complex*)malloc(sizeof(Complex)*new_size);
	memcpy(new_data+0, filter+minRadius, maxRadius*sizeof(Complex));
	memset(new_data+maxRadius, 0, (new_size-filtersize)*sizeof(Complex));
	memcpy(new_data+new_size-minRadius, filter, minRadius*sizeof(Complex));
	*paddedfilter = new_data;
	return new_size;
}

void PadData(const short *data, short **paddeddata, int length){
	int minRadius = length / 2;
	int maxRadius = length - minRadius;
	int new_size = length + maxRadius;
	short * new_data = (short *)malloc(sizeof(short)*new_size);
	memcpy(new_data + 0, data, length*sizeof(short));
	memset(new_data + length, 0, (new_size - length)*sizeof(short));
	*paddeddata = new_data;
}

void reverseArray(Complex * arr, int start, int end)
{
    while (start < end)
    {
        Complex temp = arr[start]; 
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    } 
}   
