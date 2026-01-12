import numpy as np

def gradient(h, w):
    g = np.zeros((h,w,3), dtype=np.uint8)
    for i in range(h):
        g[i,:,0] = int(255 * i / h)
        g[i,:,2] = 255
    return g
