import numpy as np
import pandas as pd

try:
    data = pd.read_csv('ap_dataset.csv')
    print("Fisier citit")
except FileNotFoundError:
    print("eroare: Fisierul cu date nu exista")
    exit()

avg_prices = data.groupby('retail_product_name')['SalePriceWithVAT'].mean().to_dict()
tabel_produse = pd.crosstab(data['id_bon'], data['retail_product_name'])

X = (tabel_produse > 0).astype(int).values
nume_produse = tabel_produse.columns.tolist()

np.random.seed(3847456)
idx = np.arange(len(X))
np.random.shuffle(idx)

limit = int(0.8 * len(X))
X_train = X[idx[:limit]]
X_test = X[idx[limit:]]

print(f"antrenare pe {len(X_train)} bonuri | testare pe {len(X_test)} bonuri")

probs = {}
probabilitati_start = {} 

numar_bonuri_train = len(X_train)
numar_total_produse = len(nume_produse)

for i in range(len(nume_produse)):
    nume_candidat = nume_produse[i]
    coloana_candidat = X_train[:, i] 
    count_candidat = np.sum(coloana_candidat)
    #laplace, se aduna la numitor cu numarul total de variante adk toate produsele
    prob_start = (count_candidat + 1) / (numar_bonuri_train + numar_total_produse) 
    probabilitati_start[nume_candidat] = prob_start
    
    probs[nume_candidat] = {}
    
    for j in range(len(nume_produse)):
        nume_item = nume_produse[j]
        coloana_item = X_train[:, j]
        
        s = np.sum((coloana_candidat == 1) & (coloana_item == 1))
        
        prob_conditionata = (s + 1) / (count_candidat + numar_total_produse)
        
        probs[nume_candidat][nume_item] = prob_conditionata

print("gata antrenare")

def ranking(cos_curent):
    scoruri = []
    for candidat in nume_produse:
        if candidat in cos_curent:
            continue
            
        scor = probabilitati_start[candidat]
        for item in cos_curent:
            p_cond = probs[candidat][item]
            scor = scor*p_cond
            
        pret = avg_prices.get(candidat, 0)
        scor_final = scor * pret
        scoruri.append((candidat, scor_final))
    
    scoruri.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scoruri]


print("\nEVALUARE LOO")

hit_counters = {1: 0, 3: 0, 5: 0}
total_incercari = 0
K_VALUES = [1, 3, 5] 

baseline_scores = []
for p in nume_produse:
    sc = probabilitati_start[p] * avg_prices.get(p, 0)
    baseline_scores.append((p, sc))

baseline_scores.sort(key=lambda x: x[1], reverse=True)
top_baseline_generic = [x[0] for x in baseline_scores]

for i in range(len(X_test)):
    row = X_test[i]
    produse_reale_in_bon = []
    for idx, val in enumerate(row):
        if val == 1:
            produse_reale_in_bon.append(nume_produse[idx])
            
    if len(produse_reale_in_bon) < 2:
        continue

    total_incercari += 1
    produs_ascuns = np.random.choice(produse_reale_in_bon)
    cos_partial = [p for p in produse_reale_in_bon if p != produs_ascuns]
    ierarhie_recomandata = ranking(cos_partial)
     
    for k in K_VALUES:
        top_k = ierarhie_recomandata[:k]
        if produs_ascuns in top_k:
            hit_counters[k] += 1


print("\nREZULTATE FINALE:")
print(f"Bonuri testate: {total_incercari}")

for k in K_VALUES:
    accuracy = hit_counters[k] / total_incercari
    print(f"Hit {k}: {accuracy:.2%} (Produsul ascuns a fost in top {k} recomandari)")


print("\nExemplu Ultimul Bon:")
print(f"Cos Partial: {cos_partial}")
print(f"Produs Ascuns (Target): {produs_ascuns}")
print(f"Recomandarile Sistemului: {ierarhie_recomandata[:5]}")