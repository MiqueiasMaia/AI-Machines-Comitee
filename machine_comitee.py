import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np 

nb = pickle.load(open('nb_classificador.sav', 'rb'))
tree = pickle.load(open('tree_classificador.sav', 'rb'))
random_forest = pickle.load(open('randomForest_classificador.sav', 'rb'))

scaler = StandardScaler()

novo = [[10000,90,1000]]
novo = np.asarray(novo)
novo = novo.reshape(-1,1)
novo = scaler.fit_transform(novo)
novo = novo.reshape(-1,3)

nb_response = nb.predict(novo)
tree_response = tree.predict(novo)
random_forest_response = random_forest.predict(novo)

print(nb_response)
print(tree_response)
print(random_forest_response)

paga = 0
nao_paga = 0

if nb_response[0] == 1:
    paga+=1
else:
    nao_paga+=1

if tree_response[0] == 1:
    paga+=1
else:
    nao_paga+=1

if random_forest_response[0] == 1:
    paga+=1
else:
    nao_paga+=1

if paga > nao_paga:
    print("Cliente pagará")
else:
    print("Cliente não pagará")


