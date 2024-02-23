import numpy as np
from scipy.interpolate import Akima1DInterpolator as interp

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


    def getWaveform(self, phase=None, numSamples=512):
        if phase == None:
            phase = np.random.random_sample()/self.sampleRate
        maxWaveformSamples = int(np.floor(self.impulseTimes[-1]*self.sampleRate))
        interpTimes = np.arange(numSamples)/self.sampleRate + phase
        maxInterpSample = min(numSamples,maxWaveformSamples)
        interpWf = self.impulseInterp(interpTimes[:maxInterpSample])
        if maxInterpSample < numSamples:
            interpWf = np.pad(interpWf, [(0,(numSamples-maxInterpSample))],
                              mode='constant',
                              constant_values=0)
        return (interpTimes, interpWf)

    # build up a 1-bit filter. Note that the flip at the end
    # is because a linear filter goes runs *backwards in time*
    # when fed to lfilter
    def _getReducedFilter(self, threshold):
        # find the last point
        grLast = np.shape(self.impulseVolts)[0]-np.argmax(np.flip(self.impulseVolts > threshold))-1
        ltLast = np.shape(self.impulseVolts)[0]-np.argmax(np.flip(self.impulseVolts < -1*threshold))-1
        last = min(grLast, ltLast)

        lastSample = int(np.ceil(self.impulseTimes[last]*self.sampleRate))
        wf = self.getWaveform(phase=0.0, numSamples=lastSample)
        gr = wf[1] > threshold
        lt = wf[1] < -1*threshold
        filt = 1*gr - 1*lt        
        return np.flip(filt[np.argmax(filt):])

        
        
