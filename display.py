import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    #unnormalize
    img = img / 2 + 0.5
    npimg = img.np()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()