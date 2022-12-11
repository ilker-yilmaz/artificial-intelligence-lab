# SORU - 3
# Data_variable_descriptions (Environmental Data) verisetine ait verileri kullanarak bir ESA mimarisi geliştiriniz.
# Geliştirmiş olduğunuz modelin doğrulama (accuracy), F1-skor ve kayıp değerlerini bulunuz.
# Bulduğunuz bu değerleri grafik üzerinde gösteriniz.
# Bunlara ek olarak tahmin işlemi gerçekleştiriniz

import pandas as pd
import numpy as np

# Data_variable_descriptions (Environmental Data) veriseti
data = pd.read_csv("datasets/Data_variable_descriptions.csv")

