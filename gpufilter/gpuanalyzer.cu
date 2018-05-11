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
  std::string filtertype;
  int top;
  int rise;
  float tau;
  float scale;
  int np;
  int newlen;
  std::string customloc;
  std::vector<std::string> datafiles;
};
//Function Definitions
void trapfiltergen(Complex*,PARAMS);//generate trap filter
void filparams(PARAMS&);//get parameters from file
void paramdefault(PARAMS&);//default parameter values in case of file error
void filtergen(Complex*,PARAMS);
void getcustomfil(Complex*,PARAMS);
void reverseArray(Complex *array,int start,int end);
int PadFilter(const Complex *, Complex **, int);
void PadData(const short *, short **,int);
double OpenDataFile(WAVEFORM **,short ***,const char *,PARAMS,int&);
void DataOutput(WAVEFORM*,short**, PARAMS, int,int,double);
void getdatafiles(PARAMS&);
void baselineshift(short*,PARAMS);
void waveshift(Complex *, Complex *,PARAMS);

//debug printing functions
void ComplexDebug(Complex*,int,char*);
void ShortDebug(short*,int,char*);

//gpu functions
void GPUConvolution(short**,Complex*,Complex*,cufftHandle,int,int,PARAMS,size_t,int);
__global__ void multiplication(Complex*,const Complex *, int, float);

int main(int argc, char* argv[]){
  int batchsize = 2048;
  //no input expected, only takes in information from parameter files
  PARAMS param;
  filparams(param);
  getdatafiles(param);
  //with parameters found now, create the filter
  Complex *filter = (Complex*)malloc(sizeof(Complex)*param.np);
  filtergen(filter,param);
  //reverseArray(filter,0,param.np);
  //now that the filters are all defined on RAM, pad them accordingly and move to GPU
  Complex *padfilter;
  param.newlen = PadFilter(filter,&padfilter,param.np);
  size_t onesize = sizeof(Complex)*param.newlen;
  size_t bulksize = sizeof(Complex)*param.newlen*batchsize;
  Complex *gpufil, *gpubulkfil;
  //now that the padded filter has been created, move it to GPU
  checkCudaErrors(cudaMalloc(&gpufil, onesize));

  checkCudaErrors(cudaMemcpy(gpufil,padfilter,onesize, cudaMemcpyHostToDevice));
  //now copy this filter for as many waveforms as will be used
  checkCudaErrors(cudaMalloc(&gpubulkfil,bulksize));
  for(int i = 0;i<batchsize;i++){
    checkCudaErrors(cudaMemcpy(&gpubulkfil[param.newlen*i],gpufil, onesize, cudaMemcpyDeviceToDevice));
  }
  //now create the FFT plan and execute it on the filter
  cufftHandle plan;
  checkCudaErrors(cufftPlan1d(&plan,param.newlen,CUFFT_C2C,batchsize));
  checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)gpubulkfil,(cufftComplex*)gpubulkfil, CUFFT_FORWARD));
  //the filter is ready to go for convolution now
  //go ahead and allocate the GPU data storage
  Complex *gpudata;
  checkCudaErrors(cudaMalloc(&gpudata,bulksize));
  //iterate once per data file
  int numfiles = param.datafiles.size();
  for(int file = 0;file<numfiles;file++){
    int numwaves = 0;
    WAVEFORM * wavedat;
    short **waveforms;
    double startval = OpenDataFile(&wavedat,&waveforms, param.datafiles.at(file).c_str(),param,numwaves);
    //now the waveforms are all read in, do the baseline shift and waveform padding
    for(int i = 0;i<numwaves;i++){
      baselineshift(waveforms[i],param);
    }
    //at this point waveforms are ready to be copied over, handle them in batches
    int numbatch=numwaves/batchsize;
    int leftover=numwaves%batchsize;
    GPUConvolution(waveforms, gpudata, gpubulkfil, plan, numbatch, leftover, param, bulksize, batchsize);  
    //now output convolved data properly
    DataOutput(wavedat,waveforms,param,file,numwaves,startval);
    free(wavedat);
    for(int i = 0;i<numwaves;i++){
      free(waveforms[i]);
    }
  }
  free(filter);
  free(padfilter);
  checkCudaErrors(cudaFree(gpufil));
  checkCudaErrors(cudaFree(gpubulkfil));
  checkCudaErrors(cudaFree(gpudata));
  checkCudaErrors(cufftDestroy(plan));
  return 0;
}

void ComplexDebug(Complex *data,int length,char *filename){
  FILE *fout;
  fout = fopen(filename,"w");
  for(int i = 0;i<length;i++){
    fprintf(fout,"%f\n",data[i].x);
  }
  std::cout<<filename<<" has been printed"<<std::endl;
}

void ShortDebug(short *data,int length,char *filename){
  FILE *fout;
  fout = fopen(filename,"w");
  for(int i = 0;i<length;i++){
    fprintf(fout,"%d\n",data[i]);
  }
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
    tempwaves[i]=(short*)calloc(param.newlen,sizeof(short));
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

__global__ void multiplication(Complex *data, Complex *filter, int length, float scale){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < length){
		Complex temp;
		temp.x = (data[i].x * filter[i].x - data[i].y * filter[i].y)*(scale);
		temp.y = (data[i].x * filter[i].y + data[i].y * filter[i].x)*(scale);
		data[i] = temp;
	}
}

void GPUConvolution(short** hostwaves,Complex *gpuwaves, Complex *gpufilter,cufftHandle plan, int numbatch, int leftover, PARAMS param, size_t bulksize,int batchsize){
  checkCudaErrors(cudaSetDevice(0));
  //first make this the simplistic version, then will optimize later
  int threads = 1024;
  int blocks = 0;
  if((batchsize*param.newlen)%threads==0){
    blocks=batchsize*param.newlen/threads;
  }
  else{
    blocks=batchsize*param.newlen/threads+1;
  }
  Complex *tempwave = (Complex*)calloc(param.newlen*batchsize,sizeof(Complex));
  float scale = param.scale/((float)param.newlen);
  Complex *temp = (Complex*)calloc(param.newlen*batchsize,sizeof(Complex));
  for(int batch=0;batch<numbatch;batch++){
    //copy waveforms over
    std::cout<<batch<<" : "<<numbatch<<std::endl;
    for(int i = 0;i<batchsize;i++){
      for(int j = 0;j<param.newlen;j++){
        tempwave[i*param.newlen+j].x=hostwaves[i+batch*batchsize][j];
        tempwave[i*param.newlen+j].y=0.0;
      }
    }
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
    //now shift data over
    for(int i = 0;i<batchsize;i++){
      waveshift(&temp[i*param.newlen],&tempwave[i*param.newlen],param);
    }
    for(int i = 0;i<batchsize;i++){
      for(int j = 0;j<param.newlen;j++){
        hostwaves[i+batch*batchsize][j]=temp[i*param.newlen+j].x;
      }
    }  
  }
  //now for what is left
  if(leftover!=0){
    std::cout<<"analyzing leftover waves"<<std::endl;
    for(int i = 0;i<leftover;i++){
      for(int j = 0;j<param.newlen;j++){
        tempwave[i*param.newlen+j].x=hostwaves[i+numbatch*batchsize][j];
        tempwave[i*param.newlen+j].y=0.0f;
      }
    }
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
    //now shift data over, only the ones we care about again
    for(int i = 0;i<leftover;i++){
      waveshift(&temp[i*param.newlen],&tempwave[i*param.newlen],param);
    }
    //change the final location a bit, but everything else is the same easily enough
    for(int i = 0;i<leftover;i++){
      for(int j = 0;j<param.newlen;j++){
        hostwaves[i+numbatch*batchsize][j]=temp[i*param.newlen+j].x;
      }
    }  
  }
  free(temp);
  free(tempwave);
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
void filtergen(Complex *filter, PARAMS param){
  if(param.filtertype=="trapfilter"){//if using trapezoidal filter, call that function
    trapfiltergen(filter, param);
  }
  else{//otherwise use a custom filter from a file
    getcustomfil(filter, param);
  }
}

void getcustomfil(Complex* filter, PARAMS param){
  //open file for custom filter
  std::ifstream filin;
  filin.open(param.customloc.c_str());
  std::string line;
  if(filin.is_open()){
    int loc = 0;
    for(int i; filin>>i;){
      filter[loc].x=i;
      filter[loc].y=0.0;
      loc++;
    }
    while(loc<param.np-1){
      filter[loc].x=0.0;
      filter[loc].y=0.0;
      loc++;
    }
  }
  else{
    std::cout<<"Custom Filter File failed to open: Code Exiting"<<std::endl;
  }
  filin.close();
}

//generate the trapezoidal filter based on parameters given
void trapfiltergen(Complex *filter, PARAMS param){
  int rise = param.rise;
  int top = param.top;
  int tau = param.tau;
  int np = param.np;
	for (int i=0;i<rise;++i){
		filter[i].x=tau+i;
		filter[i].y=0;
		filter[i+rise+top].x=rise-tau-(i);
		filter[i+rise+top].y=0;
	}
	for (int i=rise;i<rise+top;++i)
	{
		filter[i].x=rise;
		filter[i].y=0;
	}
	//now pad the filter for the appropriate number of points
	for(int i = rise+rise+top;i<np;i++){
		filter[i].x = 0;
		filter[i].y = 0;
	}
}


//setup default parameters
void paramdefault(PARAMS& param){
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
    pfin >> param.filtertype;
    if(param.filtertype=="trapfilter"){//if the user requested trapezoidal filter get those inputs
      pfin >> param.np;
      pfin >> param.top;
      pfin >> param.rise;
      pfin >> param.tau;
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
  param.scale = 1.0f/((float)param.rise*param.tau);
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
