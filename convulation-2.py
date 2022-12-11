import numpy as np

# Giriş verisi
x = np.array([
            [[0,0,0,0,0,0,0],[0,0,0,1,0,2,0],[0,1,0,2,0,1,0],[0,1,0,2,2,0,0],[0,2,0,0,2,0,0],[0,2,1,2,2,0,0],[0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,2,-1,1,0,0,0],[0,1,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    ])

# Konvülasyon katmanları
w0 = np.array([[[-1, 0, 1],[0, 0, 1],[1, -1, 1]],
               [[-1, 0, 1],[1, -1, 1],[0, 1, 0]],
               [[-1, 1, 1],[1, 1, 0],[0, -1, 0]]])

w1 = np.array([[[0, 1, -1],[0, -1, 0],[0, -1, 1]],
               [[-1, 0, 0],[1, -1, 0],[1, -1, 0]],
               [[-1, 1, -1],[0, -1, -1],[1, 0, 0]]])

# Bias değerleri
bias0 = 1
bias1 = 0

# Konvülasyon işleminin sonucu
result = np.zeros((x.shape[0], x.shape[1] - w0.shape[1] + 1, x.shape[2] - w0.shape[2] + 1, 2))

# Konvülasyon işlemini gerçekleştir
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        for k in range(result.shape[2]):
            result[i, j, k, 0] = np.sum(x[i, j:j+w0.shape[1], k:k+w0.shape[2]] * w0) + bias0
            result[i, j, k, 1] = np.sum(x[i, j:j+w1.shape[1], k:k+w1.shape[2]] * w1) + bias1

print(result)

