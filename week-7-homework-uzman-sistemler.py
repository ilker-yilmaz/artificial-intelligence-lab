# uzman sistemler
# Aşağıda diş ağrısı için belirlenmiş olan kurallar ve bu kurallara karşılık çözümler
# verilmiştir;
# Eğer diş fırçalarken diş eti kanaması olursa, diş hastalığı vardır ve diş hekimine başvur.
# Eğer diş fırçalarken uzun süreli diş eti kanaması olursa, dişeti çekilmesi vardır ve diş hekimine başvur.
# Eğer diş eti çekilmesi var ve diş kökü görünüyorsa, dolgu yaptır.
# Eğer dişte yiyecek ve içeceklerden oluşan renk değişimi varsa, dişleri temizle.
# Eğer yeni diş çıkarken morarma görünüyorsa, diş hekimine başvur.
# Eğer dişte ağrı yapmayan çürük varsa, dolgu yaptır.
# Eğer dişteki çürük ileri derecedeyse, kanal tedavisi ve dolgu yaptır

from random import choice
from experta import *

class DişAğrısı(KnowledgeEngine):
    @Rule(AS.f << Fact(diş_fırçalarken_uzun_süreli_diş_eti_kanaması='var'),
          NOT(Fact(dişeti_çekilmesi='var')))
    def dişeti_çekilmesi(self, f):
        print("dişeti çekilmesi vardır ve diş hekimine başvur.")
        self.declare(Fact(dişeti_çekilmesi='var'))
        self.retract(f)

    @Rule(AS.f << Fact(dişeti_çekilmesi='var'),
          Fact(diş_kökü_görünüyorsa='var'))
    def dolgu_yaptır(self, f):
        print("dolgu yaptır.")
        self.retract(f)

    @Rule(AS.f << Fact(diş_fırçalarken_uzun_süreli_diş_eti_kanaması='var'),
          Fact(dişeti_çekilmesi='var'),
          Fact(diş_kökü_görünüyorsa='var'))
    def dolgu_yaptır(self, f):
        print("dolgu yaptır.")
        self.retract(f)

    @Rule(AS.f << Fact(diş_fırçalarken_uzun_süreli_diş_eti_kanaması='var'),
          Fact(dişeti_çekilmesi='var'),
          Fact(diş_kökü_görünüyorsa='var'),
          Fact(dişte_yiyecek_ve_içeceklerden_oluşan_renk_değişimi='var'))
    def dişleri_temizle(self, f):
        print("dişleri temizle.")
        self.retract(f)

    @Rule(AS.f << Fact(diş_fırçalarken_uzun_süreli_diş_eti_kanaması='var'),
          Fact(dişeti_çekilmesi='var'),
          Fact(diş_kökü_görünüyorsa='var'),
          Fact(dişte_yiyecek_ve_içeceklerden_oluşan_renk_değişimi='var'),
          Fact(yeni_diş_çıkarken_morarma
                ='var'))
    def diş_hekimine_başvur(self, f):

        print("diş hekimine başvur.")
        self.retract(f)


uzman_sistem = DişAğrısı()
uzman_sistem.reset()
uzman_sistem.declare(Fact(diş_fırçalarken_uzun_süreli_diş_eti_kanaması='var'))
uzman_sistem.run()


class Işık(Fact):
    """Trafik ışıkları için bilgiler"""
    pass

class KarşıdanKarşıyaGeçme(KnowledgeEngine):
    """Trafik ışıkları için kurallar"""
    @Rule(Işık(renk='kırmızı'))
    def kırmızı(self):
        print("bekleyiniz")

    @Rule(Işık(renk='yeşil'))
    def yeşil(self):
        print("Geç")

    @Rule(Işık(renk='sarı'))
    def sarı(self):
        print("dikkatli geç")


uzman = KarşıdanKarşıyaGeçme()
uzman.reset()
uzman.declare(Işık(renk=choice(['sarı', 'kırmızı', 'yeşil'])))
uzman.run()
