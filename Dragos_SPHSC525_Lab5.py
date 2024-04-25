import numpy as np

def makeGrating(size, sf, ori, phase, contrast):
    x, y = np.meshgrid(np.linspace(0, size, size), np.linspace(0,size,size))
    center_value = size//2

    gradient = np.sin(np.deg2rad(ori))*x + np.cos(np.deg2rad(ori))*y

    grating = np.sin((2*np.pi * gradient) / sf) + np.deg2rad(phase)

    grating *= contrast

    return grating