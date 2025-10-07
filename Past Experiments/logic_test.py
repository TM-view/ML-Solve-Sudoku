import numpy as np

data = np.load("test_data.npy")
print(data.shape)  # (1000000, 2, 9, 9)
print(data[0, 0])  # ตัวอย่างบอร์ดที่เว้นช่อง
print(data[0, 1])  # ตัวอย่างเฉลย
