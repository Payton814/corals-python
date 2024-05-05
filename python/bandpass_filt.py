import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.interpolate import Akima1DInterpolator as interp

sos = signal.butter(20, (0.1, 0.85), 'bandpass', fs = 3, output='sos')
'''w, h = signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()

f = np.arange(0, 3, 1/100)
print(f)
f_wf = []
bp_min = 0.2
bp_max = 0.8
for i in f:
    if i > bp_min and i < bp_max:
        f_wf.append(1)
    else:
        f_wf.append(0)

plt.plot(f, f_wf)
plt.show()

s = fft.ifft(f_wf)
print(len(s))
plt.plot(np.arange(len(s)), s.real)
plt.show()'''

fn="coralsLPDA_impResponse.txt"
impResp = np.loadtxt(fn, delimiter='\t')
t, v = np.transpose(impResp)
maxWaveformSamples = int(np.floor(t[-1]*3))
interpTimes = np.arange(512)/3
maxInterpSample = min(512,maxWaveformSamples)
impulseInterp = interp(t, v)
interpWf = impulseInterp(interpTimes[:maxInterpSample])

if maxInterpSample < 512:
            interpWf = np.pad(interpWf, [(0,(512-maxInterpSample))],
                              mode='constant',
                              constant_values=0)
fftv  = np.fft.fft(interpWf)
freq = np.fft.fftfreq(len(interpWf), 1/3)

plt.plot(freq[:int(np.floor(len(freq)/2))], abs(fftv[:int(np.floor(len(freq)/2))]))
plt.show()


plt.plot(interpTimes, interpWf)
plt.title('signal')
plt.show()
noise = np.random.normal(0, .01, len(interpTimes))
std1 = np.std(noise)
print(std1)
filt_noise = signal.sosfilt(sos, noise)
std2 = np.std(filt_noise)
print(std2)
print(std2/std1)

plt.plot(interpTimes, interpWf+noise)
plt.title('noise + signal')
plt.show()
filt_sig = signal.sosfilt(sos, noise+interpWf)
plt.plot(interpTimes, noise)
plt.title('noise')
plt.show()
plt.plot(interpTimes, filt_sig)
plt.title('filtered sig + noise')
plt.show()
plt.plot(interpTimes, signal.sosfilt(sos, noise))
plt.title('filtered noise')
plt.show()
plt.plot(interpTimes, signal.sosfilt(sos, interpWf))
plt.title('filtered signal')
plt.show()


plt.hist(noise, bins=20)
plt.title('noise')
plt.show()

plt.hist(signal.sosfilt(sos, noise), range = (-0.03, 0.03), bins=20)
plt.title('filtered noise')
plt.show()