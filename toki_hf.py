import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np


oldalak_szama = 980
hubak_szama = 674
egy_oldalon_hibak_szama_legalabb = 2


ks = np.arange(50)

pmf_poisson_1 = st.poisson.pmf(ks, hubak_szama/oldalak_szama)
plt.bar(ks, pmf_poisson_1, label="Poisson elso feladat", alpha=0.8)
plt.legend()
plt.show()

kerdes1 = 1- sum(pmf_poisson_1[:egy_oldalon_hibak_szama_legalabb])
print(kerdes1)
"""
"""
hibak_szama_pontosan = 1
kerdes2 = kerdes1 * pmf_poisson_1[hibak_szama_pontosan]

print(kerdes2)
"""
"""
vesszo_hiba_arany = 3/5
vesszo_hiba = 0
nem_vesszo_hiba_legalabb = 3

pmf_poisson_2 = st.poisson.pmf(ks, vesszo_hiba_arany*hubak_szama/oldalak_szama)
pmf_poisson_3 = st.poisson.pmf(ks, (1-vesszo_hiba_arany)*hubak_szama/oldalak_szama)

vesszo_hiba_esely = pmf_poisson_2[vesszo_hiba]
nem_vesszo_hiba_esely = 1-sum(pmf_poisson_3[:nem_vesszo_hiba_legalabb])
kerdes3 = vesszo_hiba_esely*nem_vesszo_hiba_esely
print(kerdes3)
"""
"""
max_keses_masodpercben = 130*60
hivasok_masodpercenkent = 2.7/60/60


np.arange(5)
chance = 0
for i in range(max_keses_masodpercben):
    pmf_poisson = st.poisson.pmf(ks, (hivasok_masodpercenkent*i))
    chance += ((1-pmf_poisson[0]))/max_keses_masodpercben

kerdes4 = 1-(chance)
print(kerdes4)