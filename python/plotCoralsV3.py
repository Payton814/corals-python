from coralsV2 import CORALS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter

if __name__ == "__main__":
    th = 0.0175
    c = CORALS(fn="corals_halfscale_impulse_corr.txt", th=th)
    numSums = np.sum(np.abs(c.filter))
    print(numSums)
    wf = c.getWaveform()  
    wf_pp = (np.max(wf[1])-np.min(wf[1]))/2
    print(wf_pp)

    RMS = np.linspace(0.025, 0.1, 10)
    #RMS = [0.05]
    #SNR_improvement = []
    #SNR_arr = []
    #SNR_out = []
    runs = [] ## I want to run this at each SNR value some number (say 100) times to get a scatter plot around each SNR
    for iii in range(50):
        print("On run:", iii)
        SNR_arr = []
        SNR_improvement = []
        SNR_out = []
        for i in range(len(RMS)):
            #print("Matched filter has:", numSums, "terms")

            ## len(wf) = 3, wf[0] = interpTimes, wf[1] = interpWf, wf[2] = interpNoisyWf
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
            #wf_pp = (np.max(wf[1][40:120])-np.min(wf[1][40:120]))/2
            #mf_pp = (np.max(mf[269:277])-np.min(mf[269:277]))/2
            wf_pp = (np.max(wf[1])-np.min(wf[1]))/2
            mf_pp = (np.max(mf)-np.min(mf))/2
            #print("Original waveform has peak-to-peak of:", wf_pp)
            #print("Original SNR value:", wf_pp/RMS[i])
            #print("Matched filter has peak-to-peak of:", mf_pp)
            #print("The output SNR is:", mf_pp/mf_NoiseRMS)
            #print("Noise increase is:", np.sqrt(numSums))
            #print("SNR improvement is:", (mf_pp/(wf_pp)))
            #print("SNR improvement is:", (mf_pp*RMS[i]/(wf_pp*mf_NoiseRMS)))
            improve = mf_pp*RMS[i]/(wf_pp*mf_NoiseRMS)
            SNR_arr.append(wf_pp/RMS[i])
            SNR_improvement.append(improve)
            SNR_out.append(mf_pp/mf_NoiseRMS)
        
        SNR_arr = np.array(SNR_arr)
        SNR_improvement = np.array(SNR_improvement)
        SNR_out = np.array(SNR_out)
        runs.append(SNR_out)

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
    plt.show()



    run_std = []
    mean = []
    for i in range(len(runs[0])):
        std = np.std(np.transpose(runs)[i])
        mu = np.mean(np.transpose(runs)[i])
        run_std.append(std)
        mean.append(mu)
        
    ## I expect the linearity in improvement to be around input SNR of 1
    ## So i want to find the position where that happens
    iter = 0
    while np.flip(SNR_arr)[iter] < 1:
        iter = iter + 1
    if abs(np.flip(SNR_arr)[iter-1] - 1) < abs(np.flip(SNR_arr)[iter] - 1):
        iter = iter - 1

    a, b = np.polyfit(np.flip(SNR_arr)[iter:], np.flip(mean)[iter:], 1)


    for i in range(len(runs)):
        #plt.scatter(np.flip(SNR_arr), np.flip(runs[i]))
        plt.plot(np.flip(SNR_arr), np.flip(mean), marker = 'o')
        plt.errorbar(np.flip(SNR_arr), np.flip(mean), yerr = np.flip(run_std), fmt = 'o')

    plt.plot(np.flip(SNR_arr), a*np.flip(SNR_arr)+b, label = str(a) + 'x + ' + str(b), color = 'red')
    plt.xlabel('Input SNR')
    plt.ylabel('Output SNR')
    plt.title('SNR improvement using matched filter')
    plt.legend()
    plt.show()

    plt.plot(np.flip(SNR_arr), np.flip(run_std))
    #plt.plot(np.flip(SNR_arr)[iter:], a*np.flip(SNR_arr)[iter:]+b, label = str(a) + 'x + ' + str(b))
    plt.show()

    slope = []
    num_non0 = []
    for i in range(len(SNR_arr)):
        slope.append(a)
        num_non0.append(numSums)

    df = pd.DataFrame(np.flip(SNR_arr), columns = ['SNR Input'])
    df.insert(1, "mean SNR out", np.flip(mean))
    df.insert(2, "std SNR out", np.flip(run_std))
    df.insert(3, "fit slope", np.array(slope))
    df.insert(4, "Number of Sums", np.array(num_non0))
    #df.to_csv("SNR_improve_" + str(th) + "th_" + str(numSums) + "numSums_fullScale.csv")
