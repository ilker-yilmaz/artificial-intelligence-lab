import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from skfuzzy import control as ctrl

# sistemin giriş parametreleri
# bulaşık miktarı
# kirlilik derecesi
# bulaşık cinsi

# sistemin çıkış parametreleri
# yıkama zamanı
# deterjan miktarı
# su sıcaklığı
# üst sepet pompa devri
# alt sepet pompa devri

# bağımsız değişkenlerimiz
bulasik_miktari = ctrl.Antecedent(np.arange(0, 26, 1), 'bulasik_miktari')
kirlik_derecesi = ctrl.Antecedent(np.arange(0, 11, 1), 'kirlik_derecesi')
bulasik_cinsi = ctrl.Antecedent(np.arange(0, 11, 1), 'bulasik_cinsi')

# bağımlı değişkenimiz
yikama_zamani = ctrl.Consequent(np.arange(0, 26, 1), 'yikama_zamani')
deterjan_miktari = ctrl.Consequent(np.arange(0, 26, 1), 'deterjan_miktari')
su_sicakligi = ctrl.Consequent(np.arange(0, 26, 1), 'su_sicakligi')
ust_sepet_pompa_devri = ctrl.Consequent(np.arange(0, 26, 1), 'ust_sepet_pompa_devri')
alt_sepet_pompa_devri = ctrl.Consequent(np.arange(0, 26, 1), 'alt_sepet_pompa_devri')

# bağımsız değişkenlerimizin üyelik fonksiyonları
bulasik_miktari['az'] = fuzz.trimf(bulasik_miktari.universe, [0, 0, 5])
bulasik_miktari['orta'] = fuzz.trimf(bulasik_miktari.universe, [0, 5, 10])
bulasik_miktari['cok'] = fuzz.trimf(bulasik_miktari.universe, [5, 10, 10])

kirlik_derecesi['az kirli'] = fuzz.trimf(kirlik_derecesi.universe, [0, 0, 5])
kirlik_derecesi['orta kirli'] = fuzz.trimf(kirlik_derecesi.universe, [0, 5, 10])
kirlik_derecesi['cok kirli'] = fuzz.trimf(kirlik_derecesi.universe, [5, 10, 10])

bulasik_cinsi['hassas'] = fuzz.trimf(bulasik_cinsi.universe, [0, 0, 5])
bulasik_cinsi['karma'] = fuzz.trimf(bulasik_cinsi.universe, [0, 5, 10])
bulasik_cinsi['guclu'] = fuzz.trimf(bulasik_cinsi.universe, [5, 10, 10])

# bağımlı değişkenimizin üyelik fonksiyonları
yikama_zamani['cok kısa'] = fuzz.trimf(yikama_zamani.universe, [0, 0, 5])
yikama_zamani['kısa'] = fuzz.trimf(yikama_zamani.universe, [0, 5, 10])
yikama_zamani['orta'] = fuzz.trimf(yikama_zamani.universe, [5, 10, 15])
yikama_zamani['uzun'] = fuzz.trimf(yikama_zamani.universe, [10, 15, 20])
yikama_zamani['çok uzun'] = fuzz.trimf(yikama_zamani.universe, [15, 20, 20])

deterjan_miktari['çok az'] = fuzz.trimf(deterjan_miktari.universe, [0, 0, 5])
deterjan_miktari['az'] = fuzz.trimf(deterjan_miktari.universe, [0, 5, 10])
deterjan_miktari['normal'] = fuzz.trimf(deterjan_miktari.universe, [5, 10, 15])
deterjan_miktari['cok'] = fuzz.trimf(deterjan_miktari.universe, [10, 15, 20])
deterjan_miktari['cok fazla'] = fuzz.trimf(deterjan_miktari.universe, [15, 20, 20])


su_sicakligi['düşük'] = fuzz.trimf(su_sicakligi.universe, [0, 0, 5])
su_sicakligi['normal'] = fuzz.trimf(su_sicakligi.universe, [0, 5, 10])
su_sicakligi['yüksek'] = fuzz.trimf(su_sicakligi.universe, [5, 10, 10])

ust_sepet_pompa_devri['çok düşük'] = fuzz.trimf(ust_sepet_pompa_devri.universe, [0, 0, 5])
ust_sepet_pompa_devri['düşük'] = fuzz.trimf(ust_sepet_pompa_devri.universe, [0, 5, 10])
ust_sepet_pompa_devri['orta'] = fuzz.trimf(ust_sepet_pompa_devri.universe, [5, 10, 15])
ust_sepet_pompa_devri['yüksek'] = fuzz.trimf(ust_sepet_pompa_devri.universe, [10, 15, 20])
ust_sepet_pompa_devri['çok yüksek'] = fuzz.trimf(ust_sepet_pompa_devri.universe, [15, 20, 20])

alt_sepet_pompa_devri['çok düşük'] = fuzz.trimf(alt_sepet_pompa_devri.universe, [0, 0, 5])
alt_sepet_pompa_devri['düşük'] = fuzz.trimf(alt_sepet_pompa_devri.universe, [0, 5, 10])
alt_sepet_pompa_devri['orta'] = fuzz.trimf(alt_sepet_pompa_devri.universe, [5, 10, 15])
alt_sepet_pompa_devri['yüksek'] = fuzz.trimf(alt_sepet_pompa_devri.universe, [10, 15, 20])
alt_sepet_pompa_devri['çok yüksek'] = fuzz.trimf(alt_sepet_pompa_devri.universe, [15, 20, 20])


# kurallarımızı oluşturuyoruz
rule1 = ctrl.Rule(bulasik_miktari['az'] & kirlik_derecesi['az kirli'] & bulasik_cinsi['hassas'],
                    yikama_zamani['cok kısa'] & deterjan_miktari['çok az'] & su_sicakligi['düşük'] & ust_sepet_pompa_devri['çok düşük'] & alt_sepet_pompa_devri['çok düşük'])
rule2 = ctrl.Rule(bulasik_miktari['az'] & kirlik_derecesi['az kirli'] & bulasik_cinsi['karma'],
                    yikama_zamani['cok kısa'] & deterjan_miktari['çok az'] & su_sicakligi['düşük'] & ust_sepet_pompa_devri['çok düşük'] & alt_sepet_pompa_devri['çok düşük'])
rule3 = ctrl.Rule(bulasik_miktari['az'] & kirlik_derecesi['az kirli'] & bulasik_cinsi['guclu'],
                    yikama_zamani['kısa'] & deterjan_miktari['az'] & su_sicakligi['normal'] & ust_sepet_pompa_devri['düşük'] & alt_sepet_pompa_devri['düşük'])
rule4 = ctrl.Rule(bulasik_miktari['az'] & kirlik_derecesi['orta kirli'] & bulasik_cinsi['hassas'],
                    yikama_zamani['kısa'] & deterjan_miktari['az'] & su_sicakligi['normal'] & ust_sepet_pompa_devri['düşük'] & alt_sepet_pompa_devri['düşük'])
rule5 = ctrl.Rule(bulasik_miktari['az'] & kirlik_derecesi['orta kirli'] & bulasik_cinsi['karma'],
                    yikama_zamani['kısa'] & deterjan_miktari['az'] & su_sicakligi['normal'] & ust_sepet_pompa_devri['düşük'] & alt_sepet_pompa_devri['düşük'])
rule6 = ctrl.Rule(bulasik_miktari['az'] & kirlik_derecesi['orta kirli'] & bulasik_cinsi['guclu'],
                    yikama_zamani['orta'] & deterjan_miktari['normal'] & su_sicakligi['yüksek'] & ust_sepet_pompa_devri['orta'] & alt_sepet_pompa_devri['orta'])
rule7 = ctrl.Rule(bulasik_miktari['az'] & kirlik_derecesi['çok kirli'] & bulasik_cinsi['hassas'],
                    yikama_zamani['orta'] & deterjan_miktari['normal'] & su_sicakligi['yüksek'] & ust_sepet_pompa_devri['orta'] & alt_sepet_pompa_devri['orta'])
rule8 = ctrl.Rule(bulasik_miktari['az'] & kirlik_derecesi['çok kirli'] & bulasik_cinsi['karma'],
                    yikama_zamani['orta'] & deterjan_miktari['normal'] & su_sicakligi['yüksek'] & ust_sepet_pompa_devri['orta'] & alt_sepet_pompa_devri['orta'])
rule9 = ctrl.Rule(bulasik_miktari['az'] & kirlik_derecesi['çok kirli'] & bulasik_cinsi['guclu'],
                    yikama_zamani['uzun'] & deterjan_miktari['çok fazla'] & su_sicakligi['çok yüksek'] & ust_sepet_pompa_devri['yüksek'] & alt_sepet_pompa_devri['yüksek'])
rule10 = ctrl.Rule(bulasik_miktari['orta'] & kirlik_derecesi['az kirli'] & bulasik_cinsi['hassas'],
                    yikama_zamani['kısa'] & deterjan_miktari['az'] & su_sicakligi['normal'] & ust_sepet_pompa_devri['düşük'] & alt_sepet_pompa_devri['düşük'])
rule11 = ctrl.Rule(bulasik_miktari['orta'] & kirlik_derecesi['az kirli'] & bulasik_cinsi['karma'],
                    yikama_zamani['kısa'] & deterjan_miktari['az'] & su_sicakligi['normal'] & ust_sepet_pompa_devri['düşük'] & alt_sepet_pompa_devri['düşük'])
rule12 = ctrl.Rule(bulasik_miktari['orta'] & kirlik_derecesi['az kirli'] & bulasik_cinsi['guclu'],
                    yikama_zamani['orta'] & deterjan_miktari['normal'] & su_sicakligi['yüksek'] & ust_sepet_pompa_devri['orta'] & alt_sepet_pompa_devri['orta'])
rule13 = ctrl.Rule(bulasik_miktari['orta'] & kirlik_derecesi['orta kirli'] & bulasik_cinsi['hassas'],
                    yikama_zamani['orta'] & deterjan_miktari['normal'] & su_sicakligi['yüksek'] & ust_sepet_pompa_devri['orta'] & alt_sepet_pompa_devri['orta'])
rule14 = ctrl.Rule(bulasik_miktari['orta'] & kirlik_derecesi['orta kirli'] & bulasik_cinsi['karma'],
                    yikama_zamani['orta'] & deterjan_miktari['normal'] & su_sicakligi['yüksek'] & ust_sepet_pompa_devri['orta'] & alt_sepet_pompa_devri['orta'])
rule15 = ctrl.Rule(bulasik_miktari['orta'] & kirlik_derecesi['orta kirli'] & bulasik_cinsi['guclu'],
                    yikama_zamani['uzun'] & deterjan_miktari['çok fazla'] & su_sicakligi['çok yüksek'] & ust_sepet_pompa_devri['yüksek'] & alt_sepet_pompa_devri['yüksek'])
rule16 = ctrl.Rule(bulasik_miktari['orta'] & kirlik_derecesi['çok kirli'] & bulasik_cinsi['hassas'],
                    yikama_zamani['uzun'] & deterjan_miktari['çok fazla'] & su_sicakligi['çok yüksek'] & ust_sepet_pompa_devri['yüksek'] & alt_sepet_pompa_devri['yüksek'])
rule17 = ctrl.Rule(bulasik_miktari['orta'] & kirlik_derecesi['çok kirli'] & bulasik_cinsi['karma'],
                    yikama_zamani['uzun'] & deterjan_miktari['çok fazla'] & su_sicakligi['çok yüksek'] & ust_sepet_pompa_devri['yüksek'] & alt_sepet_pompa_devri['yüksek'])
rule18 = ctrl.Rule(bulasik_miktari['orta'] & kirlik_derecesi['çok kirli'] & bulasik_cinsi['guclu'],
                    yikama_zamani['uzun'] & deterjan_miktari['çok fazla'] & su_sicakligi['çok yüksek'] & ust_sepet_pompa_devri['yüksek'] & alt_sepet_pompa_devri['yüksek'])
rule19 = ctrl.Rule(bulasik_miktari['çok fazla'] & kirlik_derecesi['az kirli'] & bulasik_cinsi['hassas'],
                    yikama_zamani['kısa'] & deterjan_miktari['az'] & su_sicakligi['normal'] & ust_sepet_pompa_devri['düşük'] & alt_sepet_pompa_devri['düşük'])
rule20 = ctrl.Rule(bulasik_miktari['çok fazla'] & kirlik_derecesi['az kirli'] & bulasik_cinsi['karma'],
                    yikama_zamani['kısa'] & deterjan_miktari['az'] & su_sicakligi['normal'] & ust_sepet_pompa_devri['düşük'] & alt_sepet_pompa_devri['düşük'])
rule21 = ctrl.Rule(bulasik_miktari['çok fazla'] & kirlik_derecesi['az kirli'] & bulasik_cinsi['guclu'],
                    yikama_zamani['orta'] & deterjan_miktari['normal'] & su_sicakligi['yüksek'] & ust_sepet_pompa_devri['orta'] & alt_sepet_pompa_devri['orta'])
rule22 = ctrl.Rule(bulasik_miktari['çok fazla'] & kirlik_derecesi['orta kirli'] & bulasik_cinsi['hassas'],
                    yikama_zamani['orta'] & deterjan_miktari['normal'] & su_sicakligi['yüksek'] & ust_sepet_pompa_devri['orta'] & alt_sepet_pompa_devri['orta'])
rule23 = ctrl.Rule(bulasik_miktari['çok fazla'] & kirlik_derecesi['orta kirli'] & bulasik_cinsi['karma'],
                    yikama_zamani['orta'] & deterjan_miktari['normal'] & su_sicakligi['yüksek'] & ust_sepet_pompa_devri['orta'] & alt_sepet_pompa_devri['orta'])
rule24 = ctrl.Rule(bulasik_miktari['çok fazla'] & kirlik_derecesi['orta kirli'] & bulasik_cinsi['guclu'],
                    yikama_zamani['uzun'] & deterjan_miktari['çok fazla'] & su_sicakligi['çok yüksek'] & ust_sepet_pompa_devri['yüksek'] & alt_sepet_pompa_devri['yüksek'])
rule25 = ctrl.Rule(bulasik_miktari['çok fazla'] & kirlik_derecesi['çok kirli'] & bulasik_cinsi['hassas'],
                    yikama_zamani['uzun'] & deterjan_miktari['çok fazla'] & su_sicakligi['çok yüksek'] & ust_sepet_pompa_devri['yüksek'] & alt_sepet_pompa_devri['yüksek'])
rule26 = ctrl.Rule(bulasik_miktari['çok fazla'] & kirlik_derecesi['çok kirli'] & bulasik_cinsi['karma'],
                    yikama_zamani['uzun'] & deterjan_miktari['çok fazla'] & su_sicakligi['çok yüksek'] & ust_sepet_pompa_devri['yüksek'] & alt_sepet_pompa_devri['yüksek'])
rule27 = ctrl.Rule(bulasik_miktari['çok fazla'] & kirlik_derecesi['çok kirli'] & bulasik_cinsi['guclu'],
                    yikama_zamani['uzun'] & deterjan_miktari['çok fazla'] & su_sicakligi['çok yüksek'] & ust_sepet_pompa_devri['yüksek'] & alt_sepet_pompa_devri['yüksek'])


bulasik_yikama_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])
bulasik_yikama = ctrl.ControlSystemSimulation(bulasik_yikama_ctrl)

bulasik_yikama.input['bulasik_miktari'] = 62
bulasik_yikama.input['kirlik_derecesi'] = 40
bulasik_yikama.input['bulasik_cinsi'] = 88

bulasik_yikama.compute()

print(bulasik_yikama.output['yikama_zamani'])
print(bulasik_yikama.output['deterjan_miktari'])
print(bulasik_yikama.output['su_sicakligi'])
print(bulasik_yikama.output['ust_sepet_pompa_devri'])
print(bulasik_yikama.output['alt_sepet_pompa_devri'])

yikama_zamani.view(sim=bulasik_yikama)
deterjan_miktari.view(sim=bulasik_yikama)
su_sicakligi.view(sim=bulasik_yikama)
ust_sepet_pompa_devri.view(sim=bulasik_yikama)
alt_sepet_pompa_devri.view(sim=bulasik_yikama)

plt.show()

