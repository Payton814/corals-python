from coralsV2 import CORALS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter, freqs

if __name__ == "__main__":
    th = 0.02
    c = CORALS(fn="coralsLPDA_impResponse.txt", th=th)
    numSums = np.sum(np.abs(c.filter))
    wf = c.getWaveform()  
    wf_pp = (np.max(wf[1])-np.min(wf[1]))/2
    print(wf_pp)

    wf = c.getWaveform(noise_bool = False)
    plt.plot(wf[0], wf[1])
    plt.xlabel('time [ns]')
    plt.show()

    wf_fft = np.fft.fft(wf[1])
    freq = np.fft.fftfreq(wf[0].shape[-1], 1/3)
    plt.plot(freq[:int(len(freq)/2)], abs(wf_fft)[:int(len(freq)/2)])
    plt.axvline(0.15)
    plt.show()

    b, a = butter(20, (0.15, 0.75), 'bandpass', analog=True)
    w, h = freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.semilogx(w, 20 * np.log10(abs(h.imag)))
    plt.semilogx(w, 20 * np.log10(abs(h.real)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency
    plt.show()

    wf_filtered = lfilter(b, a, wf[1])
    plt.plot(wf[0], wf[1])
    plt.plot(wf[0], wf_filtered, label = 'filtered')
    plt.axvline(0.15)
    plt.legend()
    plt.show()

