import numpy as np
import pandas as pd

try:
    data = pd.read_csv('ap_dataset.csv')
    print("Fisier citit")
except FileNotFoundError:
    print("eroare: Fisierul cu date nu exista")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



###FILTRARE BONURI CRAZY SCHNITZEL###
id_bon_cu_schnitzel = data[data['retail_product_name'] == 'Crazy Schnitzel']['id_bon'].unique()
data_filtrat = data[data['id_bon'].isin(id_bon_cu_schnitzel)].copy()

print(f"Am pastrat {len(id_bon_cu_schnitzel)} bonuri")



###ADAUGARE ATRIBUTE### 
data_filtrat['data_bon'] = pd.to_datetime(data_filtrat['data_bon'])
data_filtrat['day_of_week'] = data_filtrat['data_bon'].dt.dayofweek + 1
data_filtrat['is_weekend'] = data_filtrat['day_of_week'].apply(lambda x: 1 if x >= 6 else 0)

timp_features = data_filtrat.groupby('id_bon')[['day_of_week', 'is_weekend']].first()
agregari = data_filtrat.groupby('id_bon').agg(
    total_value=('SalePriceWithVAT', 'sum'),
    cart_size=('retail_product_name', 'count')
)

#matrice produse
tabel_produse = pd.crosstab(data_filtrat['id_bon'], data_filtrat['retail_product_name'])
y = tabel_produse['Crazy Sauce'].values
y = np.where(y > 0, 1, 0) 
tabel_produse = tabel_produse.drop(columns=['Crazy Sauce'])
tabel_produse = tabel_produse.drop(columns=['Crazy Schnitzel'])

X_data = pd.concat([tabel_produse, agregari, timp_features], axis=1)
columns = X_data.columns.tolist()

X = X_data.values


###NORMALIZARE DATE###
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

np.random.seed(34857)
idx = np.arange(len(X))
np.random.shuffle(idx)

training_size = int(0.8 * len(X))

X_train = X[idx[:training_size]]
X_test = X[idx[training_size:]]
y_train = y[idx[:training_size]]
y_test = y[idx[training_size:]]

print(f"Antrenare: {X_train.shape[0]} bonuri | Testare: {X_test.shape[0]} bonuri")


def training(X, y, learning_rate, epochs):

    m, n = X.shape #m instante, n atribute
    
    w = np.random.uniform(-0.04, 0.006, size=n)
    b = 0
    
    print(f"Start antrnare: ({epochs} epochs)")
    
    for i in range(epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)

        loss =  (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        diff = y - y_pred
        
        #derivate
        dw = (1 / m) * np.dot(diff,X)
        db = (1 / m) * np.sum(diff)
        
        w = w + learning_rate * dw
        b = b + learning_rate * db
        
        if i % 100 == 0:
            print(f"epoch {i}: loss = {loss:.4f}")
            
    return w, b

def prediction(X, w, b):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    predicted_class = [1 if p > 0.5 else 0 for p in probs]
    return np.array(predicted_class), probs



###MAIN###
w, b = training(X_train, y_train, learning_rate=1, epochs=4006)
pred_class, pred_prob = prediction(X_test, w, b)

accuracy = np.mean(y_test == pred_class)

#confuzion matrix
TP = np.sum((y_test == 1) & (pred_class == 1))
TN = np.sum((y_test == 0) & (pred_class == 0))
FP = np.sum((y_test == 0) & (pred_class == 1))
FN = np.sum((y_test == 1) & (pred_class == 0))

print("\nResultate Finale:")
print(f"Acuratete: {accuracy:.2%}")
print(f"Matrice Confuzie: TP={TP}, TN={TN}, FP={FP}, FN={FN}")

print("\nCe influenteaza Crazy Sauce:")
for i in range(len(columns)):
    print(f"{columns[i]:<20}: {w[i]:.4f}")