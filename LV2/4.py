import numpy as np
import matplotlib.pyplot as plt

black = np.ones((50,50))
white = np.ones((50,50)) * 255

topRow = np.hstack((black, white))
bottomRow = np.hstack((white, black))
wholeImg = np.vstack((topRow, bottomRow))
plt.imshow(wholeImg, cmap="gray") 
plt.show()