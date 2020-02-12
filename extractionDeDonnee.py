# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd


# %%
# Lit le fichier csv (+ quelques options qui font que normalement ça va un peu plus vite)
df = pd.read_csv("train.csv",engine='c',memory_map=True,nrows=1000)

# Supprime les lignes qui contiennent une donnée manquante
df.dropna(inplace=True)

# Supprime les données dupliquées
df.drop_duplicates(inplace=True)

#Supprime les colonnes qui ne sont pas utile pour le modèle.
#Il doit y avoir un moyen de le faire directement dans la lecture du fichier mais je ne l'ai pas trouvé. Il y a le
#paramètre <names> pour la fonction read_csv qui prend la liste des colonnes à lire, mais quand je l'utilise, le résultat
#que j'obtient est bizarre. Pour l'instant ça, ça marche
df.drop(['id', 'vendor_id','dropoff_datetime','store_and_fwd_flag'],axis=1,inplace=True)

#Convertion des dates dans un format normalisé
df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)

#extraction de données
df.loc[:, "date"] = df.pickup_datetime.dt.date
df.loc[:, "time"] = df.pickup_datetime.dt.time
df.loc[:, "dayOfTheWeek"] = df.pickup_datetime.dt.dayofweek
df.loc[:, "WeekOfTheYear"] = df.pickup_datetime.dt.weekofyear


# %%


