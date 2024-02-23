from corals import CORALS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

if __name__ == "__main__":
    c = CORALS(fn="coralsHalfFiltered.txt", th=0.02)
    numSums = np.sum(np.abs(c.filter))
    print("Matched filter has:", numSums, "terms")
    wf = c.getWaveform()    
    mf = lfilter(c.filter, [1], wf[1])

    wf_pp = (np.max(wf[1])-np.min(wf[1]))/2
    mf_pp = (np.max(mf)-np.min(mf))/2
    print("Original waveform has peak-to-peak of:", wf_pp)
    print("Matched filter has peak-to-peak of:", mf_pp)
    print("Noise increase is:", np.sqrt(numSums))
    print("SNR improvement is:", (mf_pp/(wf_pp*np.sqrt(numSums))))

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
    plt.show()
