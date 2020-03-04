import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import csv


n=[]
m=[]
x=[]
y=[]

with open('train.csv', 'r') as csvfile:
    
    plots= csv.reader(csvfile, delimiter=',')
    next(plots)
    count = 0
    for row in plots:
        if ((float(row[5]) > -74.03 and float(row[5]) < -73.75 and float(row[6]) < 40.85 and float(row[6]) > 40.60) 
        and (float(row[7]) > -74.03 and float(row[7]) < -73.75 and float(row[8]) < 40.85 and float(row[8]) > 40.60)):
            x.append(float(row[5]))
            y.append(float(row[6]))

            n.append(float(row[7]))
            m.append(float(row[8]))
        count = count+1
        if (count>5000):
            break

data_pickup = []
data_dropoff = []
inc = 0

for i in x:
    h = y[inc]
    v = x[inc]

    r = m[inc]
    p = n[inc]
    data_pickup.append([v,h])
    data_dropoff.append([p,r])
    inc += 1


kmeans_pickup = KMeans(n_clusters=15, random_state=2, n_init = 10).fit(data_pickup)
kmeans_dropoff = KMeans(n_clusters=15, random_state=2, n_init = 10).fit(data_dropoff)

plt.scatter(x,y, marker='o',s=1,c=kmeans_pickup.labels_)

with open('train.csv', 'r') as csvfile, \
        open('output_1.csv', 'w+',newline='') as write_obj:
    plots= csv.reader(csvfile, delimiter=',')
    writer = csv.writer(write_obj)
    writer.writerow(['pickup_datetime', 'trip_duration', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','district_pickup','district_dropoff'])
    i=0
    next(plots)
    count = 0
    for row in plots:
        if ((float(row[5]) > -74.03 and float(row[5]) < -73.75 and float(row[6]) < 40.85 and float(row[6]) > 40.60)
        and (float(row[7]) > -74.03 and float(row[7]) < -73.75 and float(row[8]) < 40.85 and float(row[8]) > 40.60)):
            writer.writerow([row[2],row[10],row[4],row[5],row[6],row[7],row[8],kmeans_pickup.labels_[i],kmeans_dropoff.labels_[i]])
            i += 1
        count = count+1
        if (count>5000):
            break


plt.title('Pickup localisation')

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()
