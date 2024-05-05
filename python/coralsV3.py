import numpy as np
from scipy.interpolate import Akima1DInterpolator as interp
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt

class CORALS:
    def __init__(self, fn="coralsLPDA_impResponse.txt", th=0.05):
        impResp = np.loadtxt(fn, delimiter='\t')
        t, v = np.transpose(impResp)
        self.impulseTimes = t
        self.impulseVolts = v


        self.impulseInterp = interp(self.impulseTimes, self.impulseVolts)
        # this is in SAMPLES PER NANOSECOND to match the times above
        self.sampleRate = 3.0
        self.filter = self._getReducedFilter(th)


    def getWaveform(self, phase=None, numSamples=512, noise_bool = False, noise_RMS = 0, bp_filt = False, Follow_Signal = False):
        if phase == None:
            phase = np.random.random_sample()/self.sampleRate
        maxWaveformSamples = int(np.floor(self.impulseTimes[-1]*self.sampleRate))
        
        interpTimes = np.arange(numSamples)/self.sampleRate + phase
        maxInterpSample = min(numSamples,maxWaveformSamples)
        interpWf = self.impulseInterp(interpTimes[:maxInterpSample])

        if noise_bool == True:
            sos = butter(20, (0.1, 0.85), 'bandpass', fs = 3, output='sos')
            ## test_noise will hold 1000 realizations of random noise at a given RMS
            ## This is to then be passed into the matched filter to understand the filter response
            ## to the noise. This is to get the SNR value post filter
            test_noise = []
            postButter_RMS = []
            test_noise_len = 5000
            for ii in range(1000):
                noise =np.random.normal(0, noise_RMS, test_noise_len)
                filt_Noise = sosfilt(sos, noise)
                test_noise.append(filt_Noise)
                std = np.std(filt_Noise)
                postButter_RMS.append(std)
            mean_RMS = np.mean(postButter_RMS)

            if (Follow_Signal):
                plt.plot(np.arange(test_noise_len), noise)
                plt.title('Noise')
                plt.show()
                plt.hist(noise, bins = 20, range = (-0.03, 0.03))
                plt.show()
                plt.plot(np.arange(test_noise_len), filt_Noise)
                plt.title('Bandpass Filtered Noise')
                plt.show()
                plt.hist(filt_Noise, bins = 20, range = (-0.03, 0.03))
                plt.show()
            #print(mean_RMS/noise_RMS)
            #print("Input RMS:", noise_RMS)
            #print("mean post bf:", mean_RMS)

            #print("woohoo im here")
            if bp_filt == False:
                noise = np.random.normal(0, noise_RMS, maxInterpSample)
                interpNoisyWf = np.add(interpWf, noise)
                interpNoisyWf_preBP = interpNoisyWf
            else:
                noise = np.random.normal(0, noise_RMS, maxInterpSample)
                #noise = lfilter(b, a, noise)
                interpNoisyWf_preBP = np.add(interpWf, noise)
                interpNoisyWf = sosfilt(sos, interpNoisyWf_preBP)
        else:
            interpNoisyWf = interpWf


        if maxInterpSample < numSamples:
            interpWf = np.pad(interpWf, [(0,(numSamples-maxInterpSample))],
                              mode='constant',
                              constant_values=0)
            interpNoisyWf = np.pad(interpNoisyWf, [(0,(numSamples-maxInterpSample))],
                              mode='constant',
                              constant_values=0)
            if (bp_filt):
                interpNoisyWf_preBP = np.pad(interpNoisyWf_preBP, [(0,(numSamples-maxInterpSample))],
                                mode='constant',
                                constant_values=0)
            
        if (noise_bool):
            return (interpTimes, interpWf, interpNoisyWf, test_noise, mean_RMS, interpNoisyWf_preBP)
        else:
            return (interpTimes, interpWf, interpNoisyWf)

    # build up a 1-bit filter. Note that the flip at the end
    # is because a linear filter goes runs *backwards in time*
    # when fed to lfilter
    def _getReducedFilter(self, threshold):
        # find the last point
        ## Note: using np.flip(array > threshold) flips the array (so array[i] <-> array[len(array) - i - 1])
        ## Then all values in the array > threshold become True, anything below is set to False
        grLast = np.shape(self.impulseVolts)[0]-np.argmax(np.flip(self.impulseVolts > threshold))-1 ## Gets the last index value that is greater than threshold
        ltLast = np.shape(self.impulseVolts)[0]-np.argmax(np.flip(self.impulseVolts < -1*threshold))-1 ## gets the last index value that is less than threshold

        ## THe filter makes values above threshold 1, below -1, all others 0
        last = min(grLast, ltLast) ## This will be the last index that gives a value above or below the threshold

        lastSample = int(np.ceil(self.impulseTimes[last]*self.sampleRate)) 
        wf = self.getWaveform(phase=0.0, numSamples=lastSample)
        gr = wf[1] > threshold
        lt = wf[1] < -1*threshold
        #print(wf[1])
        #print(lt)
        #print(gr)
        filt = 1*gr - 1*lt        
        return np.flip(filt[np.argmax(filt):])
