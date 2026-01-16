import numpy as np
import pandas as pd

try:
    data = pd.read_csv('ap_dataset.csv')
    print("Fisier citit")
except FileNotFoundError:
    print("eroare: Fisierul cu date nu exista")
    exit()

def sigmoid(z): 
    return 1 / (1 + np.exp(-z))


### ADAUGARE ATRIBUTE ### 
data['data_bon'] = pd.to_datetime(data['data_bon'])
data['day_of_week'] = data['data_bon'].dt.dayofweek + 1
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 6 else 0)

timp_features = data.groupby('id_bon')[['day_of_week', 'is_weekend']].first()
agregari = data.groupby('id_bon').agg(
    total_value=('SalePriceWithVAT', 'sum'),
    cart_size=('retail_product_name', 'count')
)

#matrice produse
tabel_produse = pd.crosstab(data['id_bon'], data['retail_product_name'])


lista_sosuri = ["Crazy Sauce", "Cheddar Sauce", "Extra Cheddar Sauce", "Garlic Sauce", "Tomato Sauce", "Blueberry Sauce", "Spicy Sauce", "Pink Sauce"]
print(f"Lista sosuri: {lista_sosuri}")

X = pd.concat([tabel_produse, agregari, timp_features], axis=1)

np.random.seed(34857)
idx = np.arange(len(X.values))
np.random.shuffle(idx)

training_size = int(0.8 * len(X.values))
train_idx = idx[:training_size]
test_idx = idx[training_size:]

print(f"Split: {len(train_idx)} Train | {len(test_idx)} Test")

def training(X, y, learning_rate, epochs):
    m, n = X.shape 
    w = np.random.uniform(-0.04, 0.006, size=n)
    b = 0
    
    for i in range(epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)

        diff = y - y_pred
        
        dw = (1 / m) * np.dot(diff,X)
        db = (1 / m) * np.sum(diff)
        
        w = w + learning_rate * dw
        b = b + learning_rate * db
            
    return w, b



models = {} 
learning_rate = 0.5
epochs = 200
K =3

for sos in lista_sosuri:
    print(f"Antrenare model pt {sos}:")
    
    y = X[sos].values
    y = np.where(y > 0, 1, 0)

    X_minus_sos = X.drop(columns=[sos])
    
    X_curr = X_minus_sos.values

    cols_curente = X_minus_sos.columns.tolist()
    
    mean = np.mean(X_curr, axis=0)
    std = np.std(X_curr, axis=0)
    X_curr = (X_curr - mean) / std
    
    X_train = X_curr[train_idx]
    y_train = y[train_idx]
    
    w, b = training(X_train, y_train, learning_rate, epochs)
    
    models[sos] = {
        'w': w,
        'b': b,
        'mean': mean,
        'std': std,
        'cols': cols_curente
    }


###MAIN###
counts_sosuri = {}
for sos in lista_sosuri:
    y_tr = X[sos].values[train_idx]
    counts_sosuri[sos] = np.sum(y_tr > 0)

sosuri_populare_baseline = sorted(counts_sosuri, key=counts_sosuri.get, reverse=True)[:K]
print(f"\nCele mai populare {K} sosuri folosind doar datele de antrenament: {sosuri_populare_baseline}")


hits_model = 0
hits_baseline = 0
total_test = 0

for i in test_idx:
    sosuri_reale = []
    for sos in lista_sosuri:
        valoare = X.iloc[i][sos]
        if valoare > 0:
            sosuri_reale.append(sos)
    
    if len(sosuri_reale) == 0:
        continue
        
    total_test += 1
    
    probabilitati_bon = {}
    
    for sos in lista_sosuri:
        model = models[sos]
        
        row_full = X.iloc[i] 
        row_values = row_full[model['cols']].values
        
        row_norm = (row_values - model['mean']) / model['std']
        
        prob = sigmoid(np.dot(row_norm, model['w']) + model['b'])
        probabilitati_bon[sos] = prob

    recomandari_model = sorted(probabilitati_bon, key=probabilitati_bon.get, reverse=True)[:K]
    
    overlap_model = [s for s in sosuri_reale if s in recomandari_model]
    if len(overlap_model) > 0:
        hits_model += 1
        
    overlap_baseline = [s for s in sosuri_reale if s in sosuri_populare_baseline]
    if len(overlap_baseline) > 0:
        hits_baseline += 1

precision_model = hits_model / total_test

#print(hits_model)
#print(total_test)

precision_baseline = hits_baseline / total_test

print(f"Numar bonuri testate (cu cel putin un sos): {total_test}")
print("-" * 30)
print(f"Baseline Hit@{K}: {precision_baseline:.2%}")
print(f"Model ML Hit@{K}: {precision_model:.2%}")

