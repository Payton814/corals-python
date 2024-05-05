import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

fs = 10e3 
t = np.arange(0, 1, 1/fs)

noise = np.random.normal(0, 1000, len(t))

#plt.plot(t, noise)
#plt.show()
a = [1, 1, 1]
b = [1,2,3]
c = np.add(a,b)
print(c)
noise_fft = np.fft.fft(noise)
freq = np.fft.fftfreq(len(t), d = 1/fs)
envelope = hilbert(np.abs(noise_fft))
amp_env = np.abs(envelope)

plt.plot(freq, np.abs(noise_fft))
plt.plot(freq, amp_env)
plt.show()

std = np.std(noise)
mean = np.mean(noise)
print("standard deviation is:", std)
print('mean is:', mean)

plt.hist(noise, bins = 'auto')
plt.show()

