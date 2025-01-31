import numpy as np
from scipy.signal import argrelextrema


data = np.load('./data_pr/data.npy')


maxn = argrelextrema(data, np.greater)

minn = argrelextrema(data, np.less)


print("Peak index and corresponding value:")
for idx in maxn[0]:
    print(f"index: {idx}, value: {data[idx]}")


print("\nTrough index and corresponding value:")
for idx in minn[0]:
    print(f"index: {idx}, value: {data[idx]}")