from corals import CORALS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

if __name__ == "__main__":
    #RMS = np.arange(0, 0.05, 1/1000)
    RMS = [0.000001]
    SNR_improvement = []
    SNR_arr = []
    SNR_out = []
    th = 0.05
    for i in range(len(RMS)):
        #c = CORALS(fn="coralsHalfFiltered.txt", th=th)
        c = CORALS(fn="coralsLPDA_impResponse.txt", th=th)
        numSums = np.sum(np.abs(c.filter))
        print("Matched filter has:", numSums, "terms")

        ## len(wf) = 3, wf[0] = interpTimes, wf[0] = interpWf, wf[2] = interpNoisyWf
        wf = c.getWaveform(noise_bool = True, noise_RMS= RMS[i])    

        ## This next part runs all the pure noise that was generated
        ## in the getWaveform() attribute through the matched filter
        ## since the noise should be centered around 0, the std and RMS
        ## should be equivalent
        mf_noiseResponse = []
        for ii in range(1000):
            mf = lfilter(c.filter, [1], wf[3][ii]) 
            mf_noiseResponse.append(np.std(mf))
        mf_NoiseRMS = np.mean(mf_noiseResponse)

        ## Want to send the noisy waveform into the filter
        mf = lfilter(c.filter, [1], wf[2])

        ## For calculating peak-peak value of the original waveform we want the non-noisy
        wf_pp = (np.max(wf[1])-np.min(wf[1]))/2
        mf_pp = (np.max(mf)-np.min(mf))/2
        print("Original waveform has peak-to-peak of:", wf_pp)
        print("Original SNR value:", wf_pp/RMS[i])
        print("Matched filter has peak-to-peak of:", mf_pp)
        print("Noise_RMS:", mf_NoiseRMS)
        print("The output SNR is:", mf_pp/mf_NoiseRMS)
        #print("Noise increase is:", np.sqrt(numSums))
        #print("SNR improvement is:", (mf_pp/(wf_pp)))
        #print("SNR improvement is:", (mf_pp*RMS[i]/(wf_pp*mf_NoiseRMS)))
        improve = mf_pp/(wf_pp*np.sqrt(numSums))
        SNR_arr.append(wf_pp/RMS[i])
        SNR_improvement.append(improve)
        SNR_out.append(mf_pp/mf_NoiseRMS)
    
    SNR_arr = np.array(SNR_arr)
    SNR_improvement = np.array(SNR_improvement)
    SNR_out = np.array(SNR_out)


    print("length of the time", len(wf[0]))
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(wf[0],wf[1])
    # make the filter plot look better
    rf = np.flip(c.filter)
    padAfter = len(wf[0]) - len(rf) - 20
    filtToPlot = np.pad(rf, [(20, padAfter)],
                        mode='constant',
                        constant_values='0')
    axs[1].plot(wf[0], filtToPlot)
    axs[2].plot(wf[0], mf)
    axs[0].axhline(th, linestyle='--', color = 'r')
    axs[0].axhline(-th, linestyle='--', color = 'r')
    axs[2].set(xlabel='Time [ns]')
    axs[0].set(title='Matched Filter Deconvolution of Full-Scale, th = 0.05')
    plt.show()

    plt.scatter(np.flip(SNR_arr), np.flip(SNR_out))
    plt.show()
