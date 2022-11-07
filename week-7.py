# bulanık mantık
# sonuç var ise bağımlı değişken denir
# bağımlı değişkenimiz bahşiş
# bağımlı değişkeni etkileyen değişkenler ise bağımsız değişkenlerdir
# bağımsız değişkenlerimiz ise yemek fiyatı, kalite, servis
# bağımsız değişkenlerimizden bahşişin ne kadar olacağını tahmin etmeye çalışacağız

# kütüphaneler
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# bağımsız değişkenlerimiz
yemek_fiyati = ctrl.Antecedent(np.arange(0, 26, 1), 'yemek_fiyati')
kalite = ctrl.Antecedent(np.arange(0, 11, 1), 'kalite')
servis = ctrl.Antecedent(np.arange(0, 11, 1), 'servis')

# bağımlı değişkenimiz
bahsis = ctrl.Consequent(np.arange(0, 26, 1), 'bahsis')

# bağımsız değişkenlerimizin üyelik fonksiyonları
yemek_fiyati['dusuk'] = fuzz.trimf(yemek_fiyati.universe, [0, 0, 13])
yemek_fiyati['orta'] = fuzz.trimf(yemek_fiyati.universe, [0, 13, 25])
yemek_fiyati['yuksek'] = fuzz.trimf(yemek_fiyati.universe, [13, 25, 25])

kalite['dusuk'] = fuzz.trimf(kalite.universe, [0, 0, 5])
kalite['orta'] = fuzz.trimf(kalite.universe, [0, 5, 10])
kalite['yuksek'] = fuzz.trimf(kalite.universe, [5, 10, 10])

servis['dusuk'] = fuzz.trimf(servis.universe, [0, 0, 5])
servis['orta'] = fuzz.trimf(servis.universe, [0, 5, 10])
servis['yuksek'] = fuzz.trimf(servis.universe, [5, 10, 10])

# bağımlı değişkenimizin üyelik fonksiyonları
bahsis['dusuk'] = fuzz.trimf(bahsis.universe, [0, 0, 13])
bahsis['orta'] = fuzz.trimf(bahsis.universe, [0, 13, 25])
bahsis['yuksek'] = fuzz.trimf(bahsis.universe, [13, 25, 25])

# bağımsız değişkenlerimizin üyelik fonksiyonlarını görselleştirme
yemek_fiyati.view()
kalite.view()
servis.view()

# bağımlı değişkenimizin üyelik fonksiyonlarını görselleştirme
bahsis.view()

# kurallarımız
rule1 = ctrl.Rule(yemek_fiyati['dusuk'] | kalite['dusuk'] | servis['dusuk'], bahsis['dusuk'])
rule2 = ctrl.Rule(servis['orta'], bahsis['orta'])
rule3 = ctrl.Rule(yemek_fiyati['yuksek'] | kalite['yuksek'] | servis['yuksek'], bahsis['yuksek'])

# kurallarımızı kontrol etme
bahsis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
bahsis = ctrl.ControlSystemSimulation(bahsis_ctrl)

# bağımsız değişkenlerimizin değerlerini girme
bahsis.input['yemek_fiyati'] = 20
bahsis.input['kalite'] = 9.8
bahsis.input['servis'] = 9.9

# tahmin
bahsis.compute()

# tahmin sonucu
print(bahsis.output['bahsis'])
bahsis.view(sim=bahsis)

# bağımsız değişkenlerimizin üyelik fonksiyonlarını görselleştirme
yemek_fiyati.view()
kalite.view()
servis.view()


