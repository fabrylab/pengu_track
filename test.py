# import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt

import numpy as np

plt.plot(np.arange(10), np.sin(np.arange(10)))
plt.savefig("./test.png")