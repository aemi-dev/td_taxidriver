import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import csv
import pandas as pd

x=[]
y=[]


with open('train.csv', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter=',')
    next(plots)
    count = 0
    for row in plots:
        if (float(row[5]) > -74.03 and float(row[5]) < -73.75 and float(row[6]) < 40.85 and float(row[6]) > 40.60):
            x.append(float(row[5]))
            y.append(float(row[6]))
        count = count+1
        if (count>5000):
            break

data = []
inc = 0

for i in x:
    h = y[inc]
    v = x[inc]
    data.append([v,h])
    inc += 1


kmeans = KMeans(n_clusters=15, random_state=2, n_init = 10).fit(data)

plt.scatter(x,y, marker='o',s=1,c=kmeans.labels_)

plt.title('Pickup localisation')

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()
