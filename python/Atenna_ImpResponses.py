import numpy as np
import matplotlib.pyplot as plt

fullScale = "coralsLPDA_impResponse.txt"
halfScale = "corals_halfscale_impulse_corr.txt"
fullimpResp = np.loadtxt(fullScale, delimiter='\t')
t_full, v_full = np.transpose(fullimpResp)

halfimpResp = np.loadtxt(halfScale, delimiter='\t')
t_half, v_half = np.transpose(halfimpResp)

fig, axs = plt.subplots(2,1)
axs[0].plot(t_half,v_half)
axs[0].set(xlabel = 'time [ns]', title = "half-scale antenna response" )

axs[1].plot(t_full,v_full)
axs[1].set(xlabel = 'time [ns]', title = "full-scale antenna response" )
plt.tight_layout()
plt.show()
