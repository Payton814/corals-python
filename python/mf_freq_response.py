#####################################################
#
#   Code: mf_freq_response.py (Version 1)
#   
#   Purpose: This code hopefully will be simple. Its purpose is
#            to show the frequency response of the matched filter
#            given some threshold value. It will also attempt to
#            exemplify how the matched filter is created and how
#            choosing a threshold value affects which parts of the 
#            antenna response the filter is latching on to.
#
#######################################################

from coralsV3 import CORALS
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

if __name__ == "__main__":
    ## The full scale antenna response had a maximum SNR improvement at th = 0.035 and 0.075
    ## The half scale antenna response had a maximum SNR improvement at th = 0.0125 and 0.0175
    th = 0.075
    c = CORALS(fn="coralsLPDA_impResponse.txt", th=th)
    #c = CORALS(fn="corals_halfscale_impulse_corr.txt", th=th)
    numSums = np.sum(np.abs(c.filter))
    print(numSums)
    wf = c.getWaveform()  
    wf_pp = (np.max(wf[1])-np.min(wf[1]))/2

    mf = signal.lfilter(c.filter, [1], wf[1])

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(wf[0],wf[1])
    # make the filter plot look better
    rf = np.flip(c.filter)
    padAfter = len(wf[0]) - len(rf) - 20
    filtToPlot = np.pad(rf, [(20, padAfter)],
                        mode='constant',
                        constant_values='0')
    axs[1].plot(wf[0], filtToPlot)
    #axs[2].plot(wf[0], mf)
    axs[0].axhline(th, linestyle='--', color = 'r')
    axs[0].axhline(-th, linestyle='--', color = 'r')
    plt.show()

    w, mf_freqz = signal.freqz(c.filter, [1], worN = 512, fs = 3)
    plt.plot(w, 20*np.log10(np.abs(mf_freqz)))
    plt.show()

    th = 0.075
    c = CORALS(fn="coralsLPDA_impResponse.txt", th=th)
    w, mf_freqz = signal.freqz(c.filter, [1], worN = 512, fs = 3)

    th = 0.035
    c = CORALS(fn="coralsLPDA_impResponse.txt", th=th)
    w1, mf_freqz1 = signal.freqz(c.filter, [1], worN = 512, fs = 3)

    th = 0.05
    c = CORALS(fn="coralsLPDA_impResponse.txt", th=th)
    w2, mf_freqz2 = signal.freqz(c.filter, [1], worN = 512, fs = 3)

    #plt.plot(w, 20*np.log10(np.abs(mf_freqz)), label = 'th = 0.075')
    #plt.plot(w1, 20*np.log10(np.abs(mf_freqz1)), label = 'th = 0.035', color = 'g')
    plt.plot(w2, 20*np.log10(np.abs(mf_freqz2)), label = 'th = 0.05', color = 'darkorange')
    plt.axvline(0.85, linestyle='--', color = 'k')
    plt.axvline(0.1, linestyle='--', color = 'k')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('dB')
    plt.title('Matched Frequency Response')
    plt.legend()

    plt.show()


    print(c.filter)
    print(len(w))
    print(len(c.filter))
