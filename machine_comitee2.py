import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np 

nb = pickle.load(open('nb_classificador.sav', 'rb'))
tree = pickle.load(open('tree_classificador.sav', 'rb'))
random_forest = pickle.load(open('randomForest_classificador.sav', 'rb'))

scaler = StandardScaler()

novo = [[5000,90,5000]]
novo = np.asarray(novo)
novo = novo.reshape(-1,1)
novo = scaler.fit_transform(novo)
novo = novo.reshape(-1,3)

nb_response = nb.predict(novo)
tree_response = tree.predict(novo)
random_forest_response = random_forest.predict(novo)

nb_probability = nb.predict_proba(novo)
nb_confiance = nb_probability.max()
tree_probability = tree.predict_proba(novo)
tree_confiance = tree_probability.max()
random_forest_probability = random_forest.predict_proba(novo)
random_forest_confiance = random_forest_probability.max()

print(nb_confiance, nb_probability)
print(tree_confiance, tree_probability)
print(random_forest_confiance, random_forest_probability)

paga = 0
nao_paga = 0
minimal_confiance = 0.98

if nb_confiance >= minimal_confiance:
    if nb_response[0] == 1:
        paga+=1
    else:
        nao_paga+=1

if tree_confiance >= minimal_confiance:
    if tree_response[0] == 1:
        paga+=1
    else:
        nao_paga+=1

if random_forest_confiance >= minimal_confiance:
    if random_forest_response[0] == 1:
        paga+=1
    else:
        nao_paga+=1


if paga > nao_paga:
    print("Cliente pagará")
elif paga == nao_paga:
    print("Probabilidades iguais")
else:
    print("Cliente não pagará")

print(paga)
print(nao_paga)


