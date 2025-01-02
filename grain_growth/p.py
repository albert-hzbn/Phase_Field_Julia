import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("voronoi_grid.csv", delimiter=",")
print(data.shape)
plt.imshow(data)
plt.show()