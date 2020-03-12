import scipy.io.wavfile
import stft
import matplotlib.pyplot  as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy  as np
import sys,os
import seaborn as sbn
sbn.set_style("whitegrid")


def padding(x, windowSize):
    #iinitialy add the boundaries of the signal,
    #so create an array of windowSize/2 zeroes that will be added ad the top and
    #bottom of the signal
    #final singal [ 0,0,0,0,0,.....  x ..... 0,0,0,0,0,0] as many 0 as windowSize/2
    #on both sides
    zeros = np.zeros(int(windowSize/2), dtype=x.dtype)
    x = np.concatenate((zeros, x, zeros), axis=-1)
    #then
    # Pad to integer number of windowed segments
    # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
    nadd = (-(x.shape[-1]-windowSize) % int(windowSize/2)) % int(windowSize)
    zeros_shape = list(x.shape[:-1]) + [int(nadd)]
    x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
    return x


#MAIN#
#read the input file
rate, audData = scipy.io.wavfile.read("../wav/partita2-1.wav")
#extract the info
length = audData.shape[0] # this is the number of samples,
#if you divide length by the rateyou get the length in seconds /rate
channel1 = audData[:,0]#[0:length]
#convert in double format
channel1 = np.double(channel1)
#channel2 = audData[:,1]
windowSize = 8192 #length of the window to analyse with STFT
hopSize = 8192  #hopsize between windows

#pad the signal based on the windowSize
print("Padding signal...")
signal = padding(channel1, windowSize
#UNCOMMENT if you want to analyse just a piece of the entire wav
#save_wav = audData[:,0][0:length]
#scipy.io.wavfile.write("study.wav",rate,save_wav)
#window_start = 3*12288
#window_end = window_start + 12288
#save_wav = audData[:,0][window_start:window_end]
#scipy.io.wavfile.write("100th.wav",rate,save_wav)

#compute the STFT
print("STFT...")
magnitude,frequencies = stft.play(signal, rate , windowSize, hopSize)
#roll the axis
#n_samples = length/windowSize + (length%windowSize)
#print(n_samples)
#sys.exit(-1)
#now recreate the magnitude array
#columns = total signal lengt/windowsize/2
#rows = windowsize/2 +1
cols = int(len(signal)/(windowSize/2)) -1
rows = int(windowSize/2)+1
print(cols, rows)
new_array = np.zeros([cols,rows])
counter = 0
for i in range(0,cols):
    for j in range(0,rows):
        new_array[i][j] = magnitude[counter]
        counter+=1
        #print(counter)

magnitude = np.rollaxis(new_array, -1, -2)

start_time = int(windowSize/2)
stop_time  = int(magnitude.shape[-1] - windowSize/2 +1)
print(start_time, stop_time)

time = np.arange(windowSize/2, signal.shape[-1] - windowSize/2 + 1,
                 windowSize)/float(rate)

time -= (windowSize/2)/ rate
#for i in range(0,10):#
#    print(magnitude[i])
#PLOT
#x-axis for  the data

x_axis_ref = np.linspace(0, len(magnitude),len(magnitude))
#colors andfigure
colors= sbn.color_palette()
col_ref = sbn.color_palette("cubehelix", 8)
fig = plt.figure(figsize=[15,10])
ax = fig.add_subplot(111)
#axis plot
ax.plot(x_axis_ref,np.abs(magnitude))
ax.xaxis.set_tick_params(labelsize=30)
ax.yaxis.set_tick_params(labelsize=30)
ax.set_ylabel(r"Freq",fontsize=30)
ax.set_xlabel(r"ticks",fontsize=30)
#lgd = ax.legend(loc="best", fontsize=30)#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          #fancybox=True, shadow=True, ncol=3,fontsize=30)

plt.tight_layout()
plt.savefig("power_spectrum.pdf")#,bbox_extra_artists=(lgd,), bbox_inches='tight')


#plt.savefig("newCB8.png",dpi=300,bbox_extra_artists=(lgd,), bbox_inches='tight')
'''
rate, audData = scipy.io.wavfile.read("wav/toccata_fugue_dmin.wav")
length = audData.shape[0]/rate #take only the first 10 seconds
#fugue 2.30 = 190 sec
#the fugue is 6700000:22342656
channel1 = audData[:,0][6740000:22342656]
#length for one single note: 6740000:6745000 --A
wind_length = 6745000-6740000
print("window length :%d" % wind_length)
#scipy.io.wavfile.write("test.wav",rate,channel1)

#sample_length = len(audData[:,0][0:1500000])
#create various fourier spectogram
print("Computing  Custom window...")
#this must be done with stft
#stft.play(channel1)
#f256,t256,Zxx256 = stft(channel1, fs=rate,window="hann",nperseg=wind_length)

print("Computing  256 window...")
f256,t256,Zxx256 = stft(channel1, fs=rate,window="hann",nperseg=256)
print("Computing 1024 window...")
f1024,t1024,Zxx1024 = stft(channel1, fs=rate,window="hann",nperseg=1024)
print("Computing 4096 window...")
f4096,t4096,Zxx4096 = stft(channel1, fs=rate,window="hann",nperseg=4096)
print("Computing 8192 window...")
f8192,t8192,Zxx8192 = stft(channel1, fs=rate,window="hann",nperseg=8192)
'''
