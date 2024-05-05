from corals import CORALS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

if __name__ == "__main__":
    th = 0.022
    c = CORALS(fn="corals_halfscale_impulse_corr.txt", th=th)
    #c = CORALS(fn="coralsLPDA_impResponse.txt", th=th)
    numSums = np.sum(np.abs(c.filter))
    print("Matched filter has:", numSums, "terms")
    #print(c.filter)
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
    axs[0].set(title = 'Half-Scale Antenna Response')
    axs[0].axhline(th, linestyle='--', color = 'r')
    axs[0].axhline(-th, linestyle='--', color = 'r')
    # make the filter plot look better
    rf = np.flip(c.filter)
    padAfter = len(wf[0]) - len(rf) - 20
    filtToPlot = np.pad(rf, [(20, padAfter)],
                        mode='constant',
                        constant_values='0')
    axs[1].plot(wf[0], filtToPlot)
    axs[2].plot(wf[0], mf)
    plt.xlabel('Time [ns]')
    plt.ylabel('Volts [V]')
    plt.title('Half-Scale Antenna Response post Matched Filter')
    plt.tight_layout()
    plt.show()
    print(c.filter)
