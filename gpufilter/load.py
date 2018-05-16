import numpy as np
import os
import struct
import matplotlib.pyplot as plt

def help():
  print('To use load, just call the load function. Ex: load.load(/path/to/run_84.fin')
  print('This program returns the original data array and a corresponding energy array')
  print('The data are arranged in columns as follows: \n Result	EvID	Board	Channel	Timestamp	tau1  (falltime)	tau2 (risetime)	V0	t0	Residual Residual_Comparison	Energy')
  return

def load(path,num):
  numwaves = (os.stat(path).st_size-8)/7033
  size = (os.stat(path).st_size-8)
  headerformat="B,i,i,i,Q,Q,i"
  waveformat=''
  headers=[]
  waveforms=[]
  for i in range(3500):
    waveformat+='h'
  with open(path,"rb") as f:
    f.seek(8) #get past initial wave thing
    for i in range(int(num)):
      head=f.read(33)
      headers.append(struct.unpack("<ciiiqqi",head))    
      head=f.read(7000)
      waveforms.append(struct.unpack(waveformat,head))
      
    #print(formats)
  return headers,waveforms

def baselineshift(waves,pretrig):
  #first get how many waves are going to be shifted
  numwaves=len(waves)
  for i in range(numwaves):
    array = np.asarray(waves[i])
    for j in range(len(array)):
      array[j] = int(array[j])%16383
      if(array[j]>=8192):
        array[j]=array[j]-16384
    ave=np.mean(array[0:pretrig])
    array=array-ave
    waves[i]=array.tolist()
  return waves

def shiftone(waves,pretrig):
  #first get how many waves are going to be shifted
  array = np.asarray(waves)
  for j in range(len(array)):
    array[j] = int(array[j])%16383
    if(array[j]>=8192):
      array[j]=array[j]-16384
  ave=np.mean(array[0:pretrig])
  array=array-ave
  waves=array.tolist()
  return waves
    

def plotwave(waveid,path1,path2):
  #this function will open both the original and the binary file and plot a particular waveform 
  wave1=[]
  wave2=[]
  trash=[]
  waveformat=''
  for i in range(3500):
    waveformat+='h'
  with open(path1,"rb") as f:
    f.seek(8+waveid*7033) 
    trash=f.read(33)
    wave=f.read(7000)
    wave1=struct.unpack(waveformat,wave)
  with open(path2,"rb") as f:
    f.seek(8+waveid*7033)
    trash=f.read(33)
    wave=f.read(7000)
    wave2=struct.unpack(waveformat,wave)
  wave1=shiftone(wave1,900)
  plt.plot(wave1)
  plt.plot(wave2)
  plt.show()
  return
  

   
