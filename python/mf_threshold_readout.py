import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#th = ['0.005', '0.0075', '0.01', '0.0125', '0.015', '0.0175', '0.02', '0.0225', '0.025', '0.0275', '0.03', '0.0325', '0.035']
#th_arr = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035]
th_arr = np.arange(0.005, 0.105, 0.0025)
print(str(th_arr))
SNR_out = []
slope = []
num_sums = []
for i in th_arr:
    print(str(round(i, 4)))
    #df = pd.read_csv("./SNR_improve_" + str(round(i, 4)) + "th_fullScale_V4_0.1to0.85BP.csv")
    df = pd.read_csv("./SNR_improve_" + str(round(i, 4)) + "th_fullScale_V3.csv")
    SNR_out.append([np.array(df['mean SNR out']), round(i, 4)])
    slope.append(np.array(df['fit slope'][0]))
    num_sums.append(np.array(df['Number of Sums'][0]))

SNR_in = np.array(df['SNR Input'])
slope_arr = []

firstMax = np.argmax(slope)
#SecondMax = firstMax + np.argmax(slope[np.argmax(slope)+1:]) + 1

fig, axs = plt.subplots(1,2)
#for i in range(len(th_arr)):
    #axs[0].plot(SNR_in, SNR_out[i][0], label = SNR_out[i][1])
axs[0].plot(SNR_in, SNR_out[firstMax][0], label = SNR_out[firstMax][1], color = 'm')
#axs[0].plot(SNR_in, SNR_out[SecondMax][0], label = SNR_out[SecondMax][1], color = 'c')
axs[0].set(xlabel = 'Input SNR', ylabel = 'Output SNR', title = 'SNR Improvement')
axs[0].legend()
#plt.xlabel('Input SNR')
#plt.ylabel('Output SNR')
#plt.title('SNR improvement for various threshold values')

#print(slope)

axs[1].plot(th_arr, slope, label = 'slope fit')
axs[1].set(xlabel = 'mf threshold', ylabel = 'slope', title = 'Half-Scale Antenna')
axs2 = axs[1].twinx()
axs2.plot(th_arr, num_sums, color = 'r', label = 'num sums')
axs2.set(ylabel = 'Number of Nonzero coeff')
axs2.legend(loc = 'upper right')
axs[1].legend(loc = 'upper left')
axs[1].axvline(th_arr[np.argmax(slope)], linestyle='--', color = 'm')
#axs[1].axvline(th_arr[SecondMax], linestyle='--', color = 'c')
plt.show()



