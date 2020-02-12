### LES "#%%" SONT POUR LANCER DEFEINIR DES CELLULES POUR JUPYTER ###

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# create a pipeline object
# Bon, je ne suis pas bien sûr de savoir à quoi ça sert pour l'instant mais c'est dans le tuto donc pourquoi pas
pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))

#%%
# load the iris dataset and split it into train and test sets
# J'imagine qu'il va faloir adapter cette ligne pour charger les données de train.csv
X, y = load_iris(return_X_y=True)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, random_state=0) # ??? Que ce passe-t-il ici ???

#%%
# fit the whole pipeline
pipe.fit(X,y)

# %%
# we can now use it like any other estimator
accuracy_score(pipe.predict(Xtest),Ytest)