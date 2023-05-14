#-------------------------------------------------------------------------
# AUTHOR: Joseph Luna
# FILENAME: clustering.py
# SPECIFICATION: Uses K-means clustering algorithm to group several different types of data
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library
print(df)

#assign your training data to X_training feature matrix
X_training = df

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
sils = []
Ks = []

for k in range(2, 21):
     kmeans = KMeans(n_clusters=k,random_state=0)
     kmeans.fit(X_training)


     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     sils.append(silhouette_score(X_training, kmeans.labels_))
     Ks.append(k)
print(sils)
print(Ks)
#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(Ks, sils)
plt.title("k vs silhouette coefficient")
plt.xlabel("k value")
plt.ylabel("Silhouette Coefficient")
plt.show()

#read the max silhouette value index and use that k value
maxsil = Ks[sils.index(max(sils))]

#reading the test data (clusters) by using Pandas library
test_df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
labels = np.array(test_df.values).reshape(1,len(test_df))[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
