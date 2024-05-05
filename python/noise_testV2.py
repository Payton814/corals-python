from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# It takes my CPU about 50 sec to generate 1 sec of "noise".
def band_limited_noise(min_freq, max_freq, freq_step, samples=44100, samplerate=44100):
    t = np.linspace(0, samples/samplerate, samples)
    freqs = np.arange(min_freq, max_freq+1, freq_step)
    no_of_freqs = len(freqs)
    pi_2 = 2*np.pi
    phases = np.random.rand(no_of_freqs)*pi_2 # I am not sure why it is necessary to add randomness?
    signals = np.zeros(samples)
    for i in range(no_of_freqs):
        signals = signals + np.sin(pi_2*freqs[i]*t + phases[i])
    peak_value = np.max(np.abs(signals))
    signals /= peak_value
    return signals


if __name__ == '__main__':

    # CONFIGURE HERE:
    seconds = 3
    samplerate = 44100
    freq_step = 0.25
    freq = np.arange(10, 20000+1, freq_step)
    x = band_limited_noise(10, 20000, freq_step, seconds * samplerate, samplerate)
    wavfile.write("add all frequencies 3 sec freq_step 0,25.wav", 44100, x)
    plt.plot(freq, x)
    plt.show()