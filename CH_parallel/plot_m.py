import numpy as np
import matplotlib.pyplot as plt


dir = "./output" 
data = np.genfromtxt(f"{dir}/time_1000.0.dat", dtype=float)

print(data.shape)
plt.imshow(data)
plt.savefig(f"plot.png", dpi=300, bbox_inches='tight')
plt.close()
