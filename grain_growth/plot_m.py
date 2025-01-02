import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("output/time_5000.dat")
print(data.shape)
plt.imshow(data)
plt.show()