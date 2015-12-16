import numpy as np

# generate then 128 * 128 randn matrix
fname = "data/hamming/projMat.txt"
nbits = 64
projMat = np.linalg.qr(np.random.rand(128,  128), mode='complete')[0][:nbits]
np.savetxt(fname, projMat, fmt="%10.8f", delimiter=" ")
