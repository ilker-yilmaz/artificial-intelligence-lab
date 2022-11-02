# import numpy as np
#
# VIGILANCE = 0.6  # trashhold  0 - 1.0
# LEARNING_COEF = 0.5  # standard
# train = np.array([[1, 0, 0, 0, 0, 0],
#                   [1, 1, 1, 1, 1, 0],
#                   [1, 0, 1, 0, 1, 0],
#                   [0, 1, 0, 0, 1, 1],
#                   [1, 1, 1, 0, 0, 0],
#                   [0, 0, 1, 1, 1, 0],
#                   [1, 1, 1, 1, 1, 0],
#                   [1, 1, 1, 1, 1, 1]], np.float) # 8x6
#
# test = np.array([[1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 0],
#                  [1, 1, 1, 1, 0, 0],
#                  [1, 1, 1, 0, 0, 0],
#                  [1, 1, 0, 0, 0, 0],
#                  [1, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0]], np.float) # 7x6
#
# L1_neurons_cnt = len(train[0]) # burada yaptığımız şey, train dizisindeki her bir satırın uzunluğunu alıyoruz.
# L2_neurons_cnt = 1 # burada yaptığımız şey, L2 katmanında kaç nöron olacağını belirtiyoruz.
# # Init weights from the first neuron
# bottomUps = np.array([[1 / (L1_neurons_cnt + 1) for _ in range(L1_neurons_cnt)]], np.float) # burada yaptığımız şey, L1 katmanındaki nöron sayısının bir fazlası kadar ağırlık oluşturuyoruz.
# topDowns = np.array([[1 for _ in range(L1_neurons_cnt)]], np.float) # burada yaptığımız şey, L1 katmanındaki nöron sayısının bir fazlası kadar ağırlık oluşturuyoruz.
#
# for tv in train: # burada yaptığımız şey, train dizisindeki her bir satırı tv değişkenine atıyoruz.
#     print("-------")
#     print('Train vector:', tv)
#     createNewNeuron = True # burada yaptığımız şey, yeni bir nöron oluşturulup oluşturulmayacağını belirtiyoruz.
#     outputs = [bottomUps[i].dot(tv) for i in range(L2_neurons_cnt)] # burada yaptığımız şey, L1 katmanındaki nöron sayısının bir fazlası kadar ağırlık ile train dizisindeki her bir satırı çarpıyoruz.
#     counter = L2_neurons_cnt # burada yaptığımız şey, L2 katmanındaki nöron sayısını belirtiyoruz.
#     while counter > 0: # burada yaptığımız şey, L2 katmanındaki nöron sayısının bir fazlası kadar döngü oluşturuyoruz.
#         winning_output = max(outputs) # burada yaptığımız şey, L2 katmanındaki nöron sayısının bir fazlası kadar döngü oluşturuyoruz.
#         winner_neuron_idx = outputs.index(winning_output) # burada yaptığımız şey, L2 katmanındaki nöron sayısının bir fazlası kadar döngü oluşturuyoruz.
#         # NOTE!!! Sometimes there can be more than one winning neurons
#         # Then we should select them randomly. For sake of simplicity,
#         # this was not implemented for sake of simplicity
#
#         # Because `sum(tv)` can be 0 and we can not divide by zero :(
#         tv_sum = sum(tv) # burada yaptığımız şey, train dizisindeki her bir satırın toplamını belirtiyoruz.
#         if tv_sum == 0: # burada yaptığımız şey, train dizisindeki her bir satırın toplamının 0 olup olmadığını kontrol ediyoruz.
#             similarity = 0 # burada yaptığımız şey, benzerlik değerini 0 olarak belirtiyoruz.
#         else:
#             similarity = topDowns[winner_neuron_idx].dot(tv) / (sum(tv)) # burada yaptığımız şey, benzerlik değerini belirtiyoruz.
#         print(" ", topDowns[winner_neuron_idx]) # burada yaptığımız şey, L1 katmanındaki nöron sayısının bir fazlası kadar ağırlık oluşturuyoruz.
#         print("    Bottom Ups Weights:", bottomUps[winner_neuron_idx]) # burada yaptığımız şey, L1 katmanındaki nöron sayısının bir fazlası kadar ağırlık oluşturuyoruz.
#         print("    Similartiy:", similarity) # burada yaptığımız şey, benzerlik değerini belirtiyoruz.
#         if similarity >= VIGILANCE: # burada yaptığımız şey, benzerlik değerinin VIGILANCE değerinden büyük olup olmadığını kontrol ediyoruz.
#             # Found similar neuron -> update their weights
#             createNewNeuron = False # burada yaptığımız şey, yeni bir nöron oluşturulup oluşturulmayacağını belirtiyoruz.
#             new_bottom_weights = tv * topDowns[winner_neuron_idx] / (  # burada yaptığımız şey, L1 katmanındaki nöron sayısının bir fazlası kadar ağırlık oluşturuyoruz.
#                         LEARNING_COEF + tv.dot(topDowns[winner_neuron_idx]))
#             new_top_weights = tv * topDowns[winner_neuron_idx]  # burada yaptığımız şey, L1 katmanındaki nöron sayısının bir fazlası kadar ağırlık oluşturuyoruz.
#             topDowns[winner_neuron_idx] = new_top_weights # burada yaptığımız şey, L1 katmanındaki nöron sayısının bir fazlası kadar ağırlık oluşturuyoruz.
#             bottomUps[winner_neuron_idx] = new_bottom_weights # burada yaptığımız şey, L1 katmanındaki nöron sayısının bir fazlası kadar ağırlık oluşturuyoruz.
#             break
#         else:
#             # Didn't find similar neuron
#             outputs[winner_neuron_idx] = -1  # So it won't be selected in the next iteration
#             counter -= 1
#
#     if createNewNeuron:
#         print("  Creating a new new neuron")
#         new_bottom_weights = np.array([[i / (LEARNING_COEF + sum(tv)) for i in tv]], np.float)
#         new_top_weights = np.array([[i for i in tv]], np.float)
#         print("    Weights bottomUps:", new_bottom_weights)
#         print("    Weights topDowns:", new_top_weights)
#         bottomUps = np.append(bottomUps, new_bottom_weights, axis=0)
#         topDowns = np.append(topDowns, new_top_weights, axis=0)
#         L2_neurons_cnt += 1
#
# print("=====")
# print(f"Total Classes: {L2_neurons_cnt}")
# print("Center of masses")
# print(topDowns)
# for tv in test:
#     A = list(range(L2_neurons_cnt))
#     createNewNeuron = True
#     outputs = [bottomUps[i].dot(tv) for i in A]
#     winning_weight = max(outputs)
#     winner_neuron_idx = outputs.index(winning_weight)
#     print(f"Class {winner_neuron_idx} for train vector {tv}")