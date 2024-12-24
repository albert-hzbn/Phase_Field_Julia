import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("output/composition_1000.0.dat", dtype=float, delimiter=",")

plt.imshow(data)
plt.show()
