from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('main.csv')

mass_list = df['Mass'].tolist()
radius_list = df['Radius'].tolist()

x = []

for index, star_mass in enumerate(mass_list):
    temp_list = [radius_list[index], star_mass]
    x.append(temp_list)

k_means = KMeans(n_clusters = 4, init = 'k-means++', random_state = 33)

y_kmeans = k_means.fit_predict(x)

cluster1_x, cluster1_y, cluster2_x, cluster2_y, cluster3_x, cluster3_y, cluster4_x, cluster4_y = [], [], [], [], [], [], [], []

for index, data in enumerate(x):
  if y_kmeans[index] == 0:
    cluster1_x.append(data[0])
    cluster1_y.append(data[1])
  elif y_kmeans[index] == 1:
    cluster2_x.append(data[0])
    cluster2_y.append(data[1])
  elif y_kmeans[index] == 2:
    cluster3_x.append(data[0])
    cluster3_y.append(data[1])
  elif y_kmeans[index] == 3:
    cluster4_x.append(data[0])
    cluster4_y.append(data[1])

plt.figure(figsize = (15, 7))

sns.scatterplot(cluster1_x, cluster1_y, color = "yellow", label = "Cluster 1")
sns.scatterplot(cluster2_x, cluster2_y, color = "blue", label = "Cluster 2")
sns.scatterplot(cluster3_x, cluster3_y, color = "red", label = "Cluster 3")
sns.scatterplot(cluster4_x, cluster4_y, color = "green", label = "Cluster 4")

sns.scatterplot(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], color = "black", label = "Centroids", s = 100, marker = ",")

plt.title('Cluster of Stars')
plt.xlabel('Star Radius')
plt.ylabel('Star Mass')
plt.legend()
plt.gca().invert_yaxis()

plt.show()
