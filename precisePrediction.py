# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import gc
import time
import numpy as np
import pandas as pd
import threading as th

from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# %%
df = pd.read_csv('train.csv',engine='c')


# %%
pickupDatetime  = pd.to_datetime(df.pickup_datetime)
dropoffDatetime = pd.to_datetime(df.dropoff_datetime)

dfDrop = df.drop(['id','pickup_datetime','dropoff_datetime'],axis=1)

# On reccréé des colonnes de pickup en séparant les informations de la date
dfDrop['pickupHour'] = pickupDatetime.dt.hour
dfDrop['pickupDayOfWeek'] = pickupDatetime.dt.dayofweek
dfDrop['pickupWeekOfYear'] = pickupDatetime.dt.weekofyear
dfDrop['pickupDayOfYear'] = pickupDatetime.dt.dayofyear
dfDrop['pickupMonth'] = pickupDatetime.dt.month

# Pareil pour le dropoff
dfDrop['dropoffHour'] = dropoffDatetime.dt.hour
dfDrop['dropoffDayOfWeek'] = dropoffDatetime.dt.dayofweek
dfDrop['dropoffWeekOfYear'] = dropoffDatetime.dt.weekofyear
dfDrop['dropoffDayOfYear'] = dropoffDatetime.dt.dayofyear
dfDrop['dropoffMonth'] = dropoffDatetime.dt.month


# %%
plt.hist(np.log10(df.trip_duration.values))
plt.xlabel('log10(trip_duration)')
plt.show()


# %%
# Avec ce graphique, on voit que la majoritée des valeurs se situent entre 10^2 et 10^(3,5) ~= 3200
# On peut donc supprimer les autres valeurs
dfGoodDurations = dfDrop[(dfDrop.trip_duration >= 100) & (dfDrop.trip_duration < 3200)]
print(dfDrop.shape)
print(dfGoodDurations.shape)


# %%
# Beaucoup de valeur ont été enlevé
plt.hist(np.log10(df.trip_duration.values))
plt.hist(np.log10(dfGoodDurations.trip_duration.values),color='red')
plt.xlabel('log10(trip_duration)')
plt.show()


# %%
# On transforme la colonne store_and_fwd_flag pour avoir des valeurs numérique
labelEncoder = LabelEncoder()
dfGoodDurations.store_and_fwd_flag = labelEncoder.fit_transform(dfGoodDurations.store_and_fwd_flag)


# %%
# On affiche la matrice de corélation pour voir quelles valeurs sont +/- corrélées entre elles
f = plt.figure(figsize=(13,13))
plt.matshow(dfGoodDurations[1:].corr(), fignum=f.number)
plt.xticks(range(dfGoodDurations[1:].shape[1]),labels=dfGoodDurations[1:].columns,rotation=70)
plt.yticks(range(dfGoodDurations[1:].shape[1]),dfGoodDurations[1:].columns)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.show()


# %%
# On voit que trip_duration n'est pas corrélé linéairement avec beaucoup de variables. Il est donc probable que les algorithmes de regression linéaires ne donnent pas de bon résultats. On va essayer de trouver quelles variables sont intéressantes pour l'apprentissage. Nous continuons l'analyse des données pour essayer de voir quelles variables peuvent ou non apporter des informations
dfGroupByFlag = dfGoodDurations.groupby('store_and_fwd_flag')[['trip_duration']].mean()
dfGroupByFlag.reset_index(inplace=True)
plt.bar(dfGroupByFlag.store_and_fwd_flag.values,dfGroupByFlag.trip_duration.values)
plt.xlabel('store_and_fwd_flag')
plt.ylabel('mean trip duration(s)')
plt.show()
# Le graphique montre qu'il y a quand même différence notable de temps de trajet en fonction des valeurs de store_and_fwd_flag, nous gardons donc cette colonne


# %%
dfGroupByPassengerCount = dfGoodDurations.groupby(['passenger_count'])[['trip_duration']].count()
dfGroupByPassengerCount.reset_index(inplace=True)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(13,4))
ax1.set(xlabel='passenger count', ylabel='#trip record')
ax2.set(xlabel='passenger count', ylabel='log10(#trip record)')

ax1.bar(dfGroupByPassengerCount.passenger_count.values,dfGroupByPassengerCount.trip_duration.values)
ax2.bar(dfGroupByPassengerCount.passenger_count.values,np.log10(dfGroupByPassengerCount.trip_duration.values))

fig.show()


# %%
# On voit qu'il y a quelques valeurs pour 0, 8 et 9 passagers. Comme on veut prédire des trajets, le but est qu'il y ait des passagers dans les taxis, on peut donc supprimer les lignes avec 0 passager. Enfin, pour 9 et 8, il y a tellement peu d'entrées qu'elles ne sont même pas visibles sur le graphe des log(trip_record)
dfWithoutBadPCount = dfGoodDurations[(dfGoodDurations.passenger_count != 0) & (dfGoodDurations.passenger_count <= 6)]


# %%
dfDurationByPassengerCount = dfWithoutBadPCount.groupby('passenger_count')[['trip_duration']].mean()
dfDurationByPassengerCount.reset_index(inplace=True)

plt.bar(dfDurationByPassengerCount.passenger_count.values, dfDurationByPassengerCount.trip_duration.values)
plt.xlabel('passenger count')
plt.ylabel('mean trip duration(s)')
plt.show()


# %%
# Les valeurs sont assez proches, on peut donc supprimer le champs nombre de passagers qui ne semble pas apporter beaucoup d'informations
dfWithoutPCount = dfWithoutBadPCount.loc[:, dfWithoutBadPCount.columns != 'passenger_count']


# %%
# On va regarder si les informations de dates sont importantes
tripDurationByVendorID_pickupDayOfWeek = dfWithoutPCount.groupby(['vendor_id', 'pickupDayOfWeek'])[['trip_duration']].mean()
tripDurationByVendorID_pickupDayOfWeek.reset_index(inplace=True)
pickupVendor1 = tripDurationByVendorID_pickupDayOfWeek[tripDurationByVendorID_pickupDayOfWeek.vendor_id == 1]
pickupVendor2 = tripDurationByVendorID_pickupDayOfWeek[tripDurationByVendorID_pickupDayOfWeek.vendor_id == 2]

tripDurationByVendorID_dropoffDayOfWeek = dfWithoutPCount.groupby(['vendor_id', 'dropoffDayOfWeek'])[['trip_duration']].mean()
tripDurationByVendorID_dropoffDayOfWeek.reset_index(inplace=True)
dropoffVendor1 = tripDurationByVendorID_dropoffDayOfWeek[tripDurationByVendorID_dropoffDayOfWeek.vendor_id == 1]
dropoffVendor2 = tripDurationByVendorID_dropoffDayOfWeek[tripDurationByVendorID_dropoffDayOfWeek.vendor_id == 2]

fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(13,4))
ax1.set(ylabel='average trip duration', xlabel='pickup day of week')
ax2.set(xlabel='dropoff day of week')

ax1.plot(pickupVendor1.pickupDayOfWeek,pickupVendor1.trip_duration,label='vendor1')
ax1.plot(pickupVendor2.pickupDayOfWeek,pickupVendor2.trip_duration,color='red',label='vendor2')

ax2.plot(dropoffVendor1.dropoffDayOfWeek,dropoffVendor1.trip_duration)
ax2.plot(dropoffVendor2.dropoffDayOfWeek,dropoffVendor2.trip_duration,color='red')

fig.legend()
fig.show()


# %%
tripDurationByVendorID_pickupDayOfYear = dfWithoutPCount.groupby(['vendor_id', 'pickupDayOfYear'])[['trip_duration']].mean()
tripDurationByVendorID_pickupDayOfYear.reset_index(inplace=True)
pickupVendor1 = tripDurationByVendorID_pickupDayOfYear[tripDurationByVendorID_pickupDayOfYear.vendor_id == 1]
pickupVendor2 = tripDurationByVendorID_pickupDayOfYear[tripDurationByVendorID_pickupDayOfYear.vendor_id == 2]

tripDurationByVendorID_dropoffDayOfYear = dfWithoutPCount.groupby(['vendor_id', 'dropoffDayOfYear'])[['trip_duration']].mean()
tripDurationByVendorID_dropoffDayOfYear.reset_index(inplace=True)
dropoffVendor1 = tripDurationByVendorID_dropoffDayOfYear[tripDurationByVendorID_dropoffDayOfYear.vendor_id == 1]
dropoffVendor2 = tripDurationByVendorID_dropoffDayOfYear[tripDurationByVendorID_dropoffDayOfYear.vendor_id == 2]

fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(15,4))
ax1.set(ylabel='average trip duration', xlabel='pickup day of year')
ax2.set(xlabel='dropoff day of year')

ax1.plot(pickupVendor1.pickupDayOfYear,pickupVendor1.trip_duration,label='vendor1')
ax1.plot(pickupVendor2.pickupDayOfYear,pickupVendor2.trip_duration,color='red',label='vendor2')

ax2.plot(dropoffVendor1.dropoffDayOfYear,dropoffVendor1.trip_duration)
ax2.plot(dropoffVendor2.dropoffDayOfYear,dropoffVendor2.trip_duration,color='red')

fig.legend()
fig.show()


# %%
tripDurationByVendorID_pickupWeekOfYear = dfWithoutPCount.groupby(['vendor_id', 'pickupWeekOfYear'])[['trip_duration']].mean()
tripDurationByVendorID_pickupWeekOfYear.reset_index(inplace=True)
pickupVendor1 = tripDurationByVendorID_pickupWeekOfYear[tripDurationByVendorID_pickupWeekOfYear.vendor_id == 1]
pickupVendor2 = tripDurationByVendorID_pickupWeekOfYear[tripDurationByVendorID_pickupWeekOfYear.vendor_id == 2]

tripDurationByVendorID_dropoffWeekOfYear = dfWithoutPCount.groupby(['vendor_id', 'dropoffWeekOfYear'])[['trip_duration']].mean()
tripDurationByVendorID_dropoffWeekOfYear.reset_index(inplace=True)
dropoffVendor1 = tripDurationByVendorID_dropoffWeekOfYear[tripDurationByVendorID_dropoffWeekOfYear.vendor_id == 1]
dropoffVendor2 = tripDurationByVendorID_dropoffWeekOfYear[tripDurationByVendorID_dropoffWeekOfYear.vendor_id == 2]

fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(15,4))
ax1.set(ylabel='average trip duration', xlabel='pickup week of year')
ax2.set(xlabel='dropoff week day of year')

ax1.plot(pickupVendor1.pickupWeekOfYear,pickupVendor1.trip_duration,label='vendor1')
ax1.plot(pickupVendor2.pickupWeekOfYear,pickupVendor2.trip_duration,color='red',label='vendor2')

ax2.plot(dropoffVendor1.dropoffWeekOfYear,dropoffVendor1.trip_duration)
ax2.plot(dropoffVendor2.dropoffWeekOfYear,dropoffVendor2.trip_duration,color='red')

fig.legend()
fig.show()


# %%
print(dfWithoutPCount.pickupWeekOfYear.max())


# %%
# Comme il y a des valeurs étranges à la fin des graphes, ont affiche la valeur maximum de la semaine de pickup et on voit que cette valeur vaut 53 ce qui n'est pas normal, une année ayant 52 semaines. On décide donc de supprimer les entéres qui ont une semaine de l'année égale à 53
dfWithoutBadWeeks = dfWithoutPCount[dfWithoutPCount.pickupWeekOfYear < 53]

tripDurationByVendorID_goodWeeks = dfWithoutBadWeeks.groupby(['vendor_id', 'pickupWeekOfYear'])[['trip_duration']].mean()
tripDurationByVendorID_goodWeeks.reset_index(inplace=True)

pickupVendor1 = tripDurationByVendorID_goodWeeks[tripDurationByVendorID_goodWeeks.vendor_id == 1]
pickupVendor2 = tripDurationByVendorID_goodWeeks[tripDurationByVendorID_goodWeeks.vendor_id == 2]

plt.plot(pickupVendor1.pickupWeekOfYear,pickupVendor1.trip_duration,label='vendor1')
plt.plot(pickupVendor2.pickupWeekOfYear,pickupVendor2.trip_duration,color='red',label='vendor2')

plt.xlabel('pickup week of year')
plt.ylabel('average trip duration')
plt.legend()
plt.show()


# %%
tripDurationByVendorID_pickupHour = dfWithoutBadWeeks.groupby(['vendor_id', 'pickupHour'])[['trip_duration']].mean()
tripDurationByVendorID_pickupHour.reset_index(inplace=True)
pickupVendor1 = tripDurationByVendorID_pickupHour[tripDurationByVendorID_pickupHour.vendor_id == 1]
pickupVendor2 = tripDurationByVendorID_pickupHour[tripDurationByVendorID_pickupHour.vendor_id == 2]

tripDurationByVendorID_dropoffHour = dfWithoutBadWeeks.groupby(['vendor_id', 'dropoffHour'])[['trip_duration']].mean()
tripDurationByVendorID_dropoffHour.reset_index(inplace=True)
dropoffVendor1 = tripDurationByVendorID_dropoffHour[tripDurationByVendorID_dropoffHour.vendor_id == 1]
dropoffVendor2 = tripDurationByVendorID_dropoffHour[tripDurationByVendorID_dropoffHour.vendor_id == 2]

fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(15,4))
ax1.set(ylabel='average trip duration', xlabel='pickup hour')
ax2.set(xlabel='dropoff hour')

ax1.plot(pickupVendor1.pickupHour,pickupVendor1.trip_duration,label='vendor1')
ax1.plot(pickupVendor2.pickupHour,pickupVendor2.trip_duration,color='red',label='vendor2')

ax2.plot(dropoffVendor1.dropoffHour,dropoffVendor1.trip_duration)
ax2.plot(dropoffVendor2.dropoffHour,dropoffVendor2.trip_duration,color='red')

fig.legend()
fig.show()


# %%
tripDurationByVendorID_pickupMonth = dfWithoutBadWeeks.groupby(['vendor_id', 'pickupMonth'])[['trip_duration']].mean()
tripDurationByVendorID_pickupMonth.reset_index(inplace=True)
pickupVendor1 = tripDurationByVendorID_pickupMonth[tripDurationByVendorID_pickupMonth.vendor_id == 1]
pickupVendor2 = tripDurationByVendorID_pickupMonth[tripDurationByVendorID_pickupMonth.vendor_id == 2]

tripDurationByVendorID_dropoffMonth = dfWithoutBadWeeks.groupby(['vendor_id', 'dropoffMonth'])[['trip_duration']].mean()
tripDurationByVendorID_dropoffMonth.reset_index(inplace=True)
dropoffVendor1 = tripDurationByVendorID_dropoffMonth[tripDurationByVendorID_dropoffMonth.vendor_id == 1]
dropoffVendor2 = tripDurationByVendorID_dropoffMonth[tripDurationByVendorID_dropoffMonth.vendor_id == 2]

fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(15,4))
ax1.set(ylabel='average trip duration', xlabel='pickup month')
ax2.set(xlabel='dropoff month')

ax1.plot(pickupVendor1.pickupMonth,pickupVendor1.trip_duration,label='vendor1')
ax1.plot(pickupVendor2.pickupMonth,pickupVendor2.trip_duration,color='red',label='vendor2')

ax2.plot(dropoffVendor1.dropoffMonth,dropoffVendor1.trip_duration)
ax2.plot(dropoffVendor2.dropoffMonth,dropoffVendor2.trip_duration,color='red')

fig.legend()
fig.show()


# %%
# On voit avec ces graphiques que globalement le vendeur 2 met un peu plus de temps que le vendeur 1 mais surtout que la date de pickup ou de dropoff donne des courbes presques parfaitement similaires, on peut donc n'en garder qu'une des deux. On choisit de garder pickup.
dropCols = ['vendor_id','dropoffDayOfWeek','dropoffWeekOfYear','dropoffDayOfYear','dropoffHour','dropoffMonth']
dfWithoutUseless = dfWithoutBadWeeks.drop(dropCols,axis=1)


# %%
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4),sharey=True)

ax1.set(xlabel='pickup latitude',ylabel='pickup longitude')
ax2.set(xlabel='dropoff latitude',ylabel='dropoff longitude')

ax1.hist2d(dfWithoutBadWeeks.pickup_latitude,dfWithoutBadWeeks.pickup_longitude,label='pickup coord')
ax2.hist2d(dfWithoutBadWeeks.dropoff_latitude,dfWithoutBadWeeks.dropoff_longitude,label='dropoff coord')
fig.show()


# %%
# On peut supprimer les lattitudes entre 40 et 42 et les longitudes entre -70 et -80
dfGood = dfWithoutBadWeeks[(dfWithoutBadWeeks.pickup_longitude >= -74.05) & (dfWithoutBadWeeks.pickup_longitude <= -73.7) & (dfWithoutBadWeeks.pickup_latitude >= 40.60) & (dfWithoutBadWeeks.pickup_latitude <= 40.9) & (dfWithoutBadWeeks.dropoff_longitude >= -74.05) & (dfWithoutBadWeeks.dropoff_longitude <= -73.7) & (dfWithoutBadWeeks.dropoff_latitude >= 40.60) & (dfWithoutBadWeeks.dropoff_latitude <= 40.9)]
print(dfWithoutBadWeeks.shape)
print(dfGood.shape)


# %%
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))

ax1.set(xlabel='pickup latitude',ylabel='pickup longitude')
ax2.set(xlabel='dropoff latitude',ylabel='dropoff longitude')

ax1.hist2d(dfGood.pickup_latitude,dfGood.pickup_longitude,label='pickup coord')
ax2.hist2d(dfGood.dropoff_latitude,dfGood.dropoff_longitude,label='dropoff coord')

fig.show()


# %%
# On a supprimé les locations trop éloignées, on va maintenant faire un cluqtering des pickups
kmeanPickup = KMeans(n_clusters=7,random_state=2,n_init=1,n_jobs=os.cpu_count()*2).fit(dfGood[['pickup_longitude','pickup_latitude']])
kmeanDropoff = KMeans(n_clusters=7,random_state=2,n_init=1,n_jobs=os.cpu_count()*2).fit(dfGood[['dropoff_longitude','dropoff_latitude']])


# %%
dfGood['clusterPickup'] = kmeanPickup.labels_

tripdurationByCluster = dfGood.groupby('clusterPickup')[['trip_duration']].mean()
tripdurationByCluster.reset_index(inplace=True)

plt.bar(tripdurationByCluster.clusterPickup.values, tripdurationByCluster.trip_duration.values)
plt.title("average trip duration per cluster")
plt.xlabel("cluster number")
plt.ylabel("average trip duration")
plt.show()


# %%
### DEFINITION DE FONCTIONS ###

def saveData():
    """
        Une fois que les données de train.csv ont été traîtée, et splitée dans des ensembles (XTrain,YTrain) (XVal,YVal), cette fonction écrit ces ensembles dans un fichier csv correspondant afin de ne pas avoir à relancer tout le notebook pour réutiliser ces ensembles mais simplement les lire depuis leurs fichiers
    """
    XTrain.to_csv('XTrain.csv')
    XVal.to_csv('XVal.csv')
    YTrain.to_csv('YTrain.csv')
    YVal.to_csv('YVal.csv')

def readData():
    """
        Lit les fichier XTrain.csv, YTrain.csv, XVal.csv, YVal.csv puis retourne leurs valeurs dans un tuple au format
        (XTrain,YTrain,XVal,YVal)

        :return: (XTrain,YTrain,XVal,YVal)
        :rtype: tuple
    """
    XTrain = pd.read_csv('XTrain.csv',engine='c',index_col=0)
    YTrain = pd.read_csv('YTrain.csv',engine='c',index_col=0).values.ravel()
    XVal = pd.read_csv('XVal.csv',engine='c',index_col=0)
    YVal = pd.read_csv('YVal.csv',engine='c',index_col=0).values.ravel()

    return (XTrain,XVal,YTrain,YVal)

def genOutputForKaggle(model,dataScaler=None):
    """
        Utilise le modèle entraîné 'model' pour prédire des temps de trajets avec les données de test.csv et convertit ces données dans un fichier sub.csv qui pourra être utilisé pour faire une publication des résultats sur Kaggle. *ATTENTION, POUR UTILISER CETTE FONCTION TOUTES LES CELLULES PRECEDENTES DOIVENT AVOIR ETE EXECUTEES*

        :param dataScaler: sklearn_scaler Le scaler utilisé pour transformer les données lors de la prédiction. Si None, les données ne seront pas transformées et seront données directement telle quelle au modèle
    """
    dfTest = pd.read_csv('test.csv',engine='c')
    dfTestId = dfTest.id
    dfTestPickupDatetime = pd.to_datetime(dfTest.pickup_datetime)
    dfTestDrop = dfTest.drop(['id','vendor_id','passenger_count','pickup_datetime'],axis=1)

    dfTestDrop['pickupHour'] = dfTestPickupDatetime.dt.hour
    dfTestDrop['pickupDayOfWeek'] = dfTestPickupDatetime.dt.dayofweek
    dfTestDrop['pickupWeekOfYear'] = dfTestPickupDatetime.dt.weekofyear
    dfTestDrop['pickupDayOfYear'] = dfTestPickupDatetime.dt.dayofyear
    dfTestDrop['pickupMonth'] = dfTestPickupDatetime.dt.month

    dfTestDrop['clusterPickup'] = kmeanPickup.predict(dfTestDrop[['pickup_longitude','pickup_latitude']])
    dfTestDrop.store_and_fwd_flag = labelEncoder.transform(dfTestDrop.store_and_fwd_flag)

    if dataScaler != None:
        preds = model.predict(dataScaler.fit_transform(dfTestDrop))
    else:
        preds = model.predict(dfTestDrop)

    result = pd.DataFrame()
    result['id'] = dfTestId
    result['trip_duration'] = preds
    result.to_csv('sub.csv',index=0)

def checkAndZeroNegativeValues(valuesList):
    valuesList[valuesList < 0] = 0

def RSMLE(y_true, y_pred):
    """
        Transforme les données négatives de y_pred en valeurs nulles puis alcule le score RSMLE  y_pred par rapport à y_true

        :param y_true: Les valeurs dont on connait le résultat
        :param y_pred: Les valeurs prédites

        :return: score RSMLE
        :rtype: float
    """
    checkAndZeroNegativeValues(y_pred)
    return np.sqrt(mean_squared_log_error(y_true,y_pred))

def printTrainingInfo(timeFit,timePred,YPreds,YVal):
    errorInSeconds = mean_absolute_error(YVal,YPreds)
    
    error = RSMLE(YVal,YPreds)

    endMsg = " t={:.2f}s (f={:.2f}s p={:.2f}s) e={:.5f} (={:.2f}s)\n".format(timeFit + timePred,timeFit,timePred,error,errorInSeconds)
    print(endMsg,end='')

def trainModelAndDisplayTrainingInfo(model,XTrain,YTrain,XVal,YVal):
    startFit = time.time()
    model.fit(XTrain,YTrain)
    endFit = time.time()

    startPred = time.time()
    preds = model.predict(XVal)
    endPred = time.time()

    printTrainingInfo(endFit-startFit,endPred-startPred,preds,YVal)


# %%
# Maintenant que l'on a des données exploitable, on va pouvoir commencer les entraînements
#X = dfGood.loc[:,dfGood.columns != 'trip_duration']
#Y = dfGood.trip_duration
#XTrain, XVal, YTrain, YVal = train_test_split(X,Y,train_size=0.8,random_state=19061996)
#saveData()
XTrain, XVal, YTrain, YVal = readData()


# %%
### TRAINING WITH RANDOM FOREST REGRESSOR ###
def train (n, d,jobs=None):
    startingMsg = "Starting n={} d={} ".format(n,d)
    print(startingMsg,end='')
    
    model = RandomForestRegressor(random_state=19061996,n_estimators=n, max_depth=d,n_jobs=jobs)

    trainModelAndDisplayTrainingInfo(model,XTrain,YTrain,XVal,YVal)

N = [10]
D = [10]

# +---------> D
# |
#\|/
# N
mask = [[1]]
print("training with {} threads".format(os.cpu_count()))
assert(len(mask) == len(N))
for i in range(len(N)):
    assert(len(mask[i]) == len(D))

for i in range(len(N)):
    for j in range (len(D)):
        if mask[i][j] == 1:
            train(N[i],D[j],os.cpu_count()*2)
    print('')


# %%
### TRAINING WITH SVR ###
# Les résultats de SVR ne sont pas indépendants de la répartition des données (c'est dit dans la doc de sklearn), il faut donc remettre X soit entre [-1,1], [0,1] ou avec une moyenne de 0 et un écart type de 1.
copyItem = 8000

scaler = preprocessing.StandardScaler()
SVRXTrain = scaler.fit_transform(XTrain[:copyItem])
SVRYTrain = YTrain[:copyItem].copy()

genOutputForKaggle(SVR(C=5314,epsilon=53.86,).fit(SVRXTrain,SVRYTrain),scaler)

#gridSearch = GridSearchCV(SVR(epsilon=157,gamma='scale'),{'C':[23435,23436,23437,23438,23439,23440,23441,23442,23443,23444]'auto']},scoring=make_scorer(RSMLE,greater_is_better=False),n_jobs=os.cpu_count()*2,error_score=0,verbose=5).fit(SVRXTrain,SVRYTrain)

#print(gridSearch.best_params_, "{:.5f}".format(-gridSearch.best_score_))


# %%
printTrainingInfo(0,0,gridSearch.predict(scaler.fit_transform(XVal)),YVal)


# %%


