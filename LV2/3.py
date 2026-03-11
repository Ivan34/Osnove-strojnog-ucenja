import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg") 
img = img[:,:,0].copy()
imgBrighter = np.clip(img * 1.95, 0, 255)


sirina = img.shape[1]
print(sirina)
cetvrtina = sirina // 4

imgSecFourth = img[:,cetvrtina:2*cetvrtina]

imgRotated = np.rot90(img, -1)

imgMirrored = np.fliplr(img)

plt.subplot(2, 2, 1)
plt.imshow(imgBrighter, cmap="gray") 

plt.subplot(2, 2, 2)
plt.imshow(imgSecFourth, cmap="gray") 

plt.subplot(2, 2, 3)
plt.imshow(imgRotated, cmap="gray") 

plt.subplot(2, 2, 4)
plt.imshow(imgMirrored, cmap="gray") 

plt.show()

