import matplotlib.pyplot as plt
import csv

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
        if (count>50000):
            break


plt.scatter(x,y, marker='o',s=1)

plt.title('Pickup localisation')

plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()
