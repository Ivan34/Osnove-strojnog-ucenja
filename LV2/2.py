import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv", delimiter=",", skiprows=1)

print("Pokus je obavljen na", data.shape[0], "ljudi")

height = data[:,1]
width = data[:,2]

plt.figure(1)
plt.scatter(height, width, c="r", alpha=0.5)
plt.xlabel("height")
plt.ylabel("width")

plt.figure(2)
plt.scatter(height[::50], width[::50],c="b")
plt.xlabel("height")
plt.ylabel("width")


print("Max height: ", height.max())
print("Min height: ", height.min())
print("Avrage height: ", height.mean())
print("\n")

maleInd = (data[:,0] == 1)
femaleInd = (data[:,0] == 0)

maleHeight = data[maleInd, 1]
femaleHeight = data[femaleInd, 1]

print("Max male height: ", maleHeight.max())
print("Min male height: ", maleHeight.min())
print("Avrage male height: ", maleHeight.mean())
print("\n")
print("Max female height: ", femaleHeight.max())
print("Min female height: ", femaleHeight.min())
print("Avrage female height: ", femaleHeight.mean())



plt.show()
