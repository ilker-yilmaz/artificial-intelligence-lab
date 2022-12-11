import numpy as np

# Örnek veriler
X = np.array([[[0,0,0],
               [0,2,1],
               [0,1,0]],
              [[0,1,0],
               [2,-1,1],
               [1,0,0]]])

w0 = np.array([[[-1,0,1],
                [0,0,0],
                [1,-1,1]],
               [[-1,1,0],
                [1,-1,1],
                [0,0,0]]])

w1 = np.array([[[1,1,1],
                [0,0,0],
                [1,1,1]],
               [[1,1,1],
                [1,1,1],
                [1,1,1]]])

b0 = 1
b1 = 1

# Konvülasyon işlemi

# İlk filtre
O0 = np.zeros((2,2))
for i in range(2):
  for j in range(2):
    for k in range(2):
      for l in range(3):
        for m in range(3):
          if i+l < 3 and j+m < 3:
            O0[i,j] += X[k,i+l,j+m] * w0[k,l,m]
    O0[i,j] += b0

# İkinci filtre
O1 = np.zeros((2,2))
for i in range(2):
  for j in range(2):
    for k in range(2):
      for l in range(3):
        for m in range(3):
          if i+l < 3 and j+m < 3:
            O1[i,j] += X[k,i+l,j+m] * w1[k,l,m]
    O1[i,j] += b1

# Sonuçlar
print(O0)
print(O1)
