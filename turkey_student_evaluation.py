# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:35:13 2021

@author: LC
"""
#Import modules
import os #provides functions for interacting with the operating system
os.chdir('C:\\Users\\LC\\Desktop\\turkey student evaluation')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 99

#Loading the dataset
df = pd.read_csv("turkiye-student-evaluation_generic.csv")
df.head()

# statistical info
df.describe()

# datatype info
df.info()

#Preprocessing the dataset

# check for null values
df.isnull().sum()

#Exploratory Data Analysis

# set new style for the graph
plt.style.use("fivethirtyeight")

sns.countplot(df['instr'])
sns.countplot(df['class'])

# find mean of questions
x_questions = df.iloc[:, 5:33]
q_mean = x_questions.mean(axis=0)
total_mean = q_mean.mean()

q_mean = q_mean.to_frame('mean')
q_mean.reset_index(level=0, inplace=True)
q_mean.head()


total_mean

plt.figure(figsize=(14,7))
sns.barplot(x='index', y='mean', data=q_mean)

#Coorelation Matrix
corr = df.corr()
plt.figure(figsize=(18,18))
sns.heatmap(corr, annot=True, cmap='coolwarm')

#Principal component analysis
X = df.iloc[:, 5:33]


from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

X_pca

# how much info we retained from the dataset
pca.explained_variance_ratio_.cumsum()[1]


#Model Training
# Kmeans clustering
from sklearn.cluster import KMeans
distortions = []
cluster_range = range(1,6)

# elbow method
for i in cluster_range:
    model = KMeans(n_clusters=i, init='k-means++', n_jobs=-1, random_state=42)
    model.fit(X_pca)
    distortions.append(model.inertia_)
    
plt.plot(cluster_range, distortions, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel('Distortions')
plt.show()

# use best cluster
model = KMeans(n_clusters=3, init='k-means++', n_jobs=-1, random_state=42)
model.fit(X_pca)
y = model.predict(X_pca)   

plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], s=50, c='red', label='cluster 1')
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], s=50, c='yellow', label='cluster 2')
plt.scatter(X_pca[y==2, 0], X_pca[y==2, 1], s=50, c='green', label='cluster 3')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:, 1], s=100, c='blue', label='centroids')
plt.title('Cluster of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()


from collections import Counter
Counter(y)

model = KMeans(n_clusters=3, init='k-means++', n_jobs=-1, random_state=42)
model.fit(X)
y = model.predict(X)

Counter(y)

# dendogram
import scipy.cluster.hierarchy as hier
dendogram = hier.dendrogram(hier.linkage(X_pca, method='ward'))
plt.title('Dendogram')
plt.xlabel("Questions")
plt.ylabel("Distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
y = model.fit_predict(X_pca)

plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], s=50, c='red', label='cluster 1')
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], s=50, c='yellow', label='cluster 2')
plt.title('Cluster of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

Counter(y)







