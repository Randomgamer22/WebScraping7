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

wcss = []

for i in range(1, 11):
    k_means = KMeans(n_clusters = i, init = 'k-means++', random_state = 33)

    k_means.fit(x)

    wcss.append(k_means.inertia_)

plt.figure(figsize = (10, 5))
sns.lineplot(range(1, 11), wcss, marker = 'o', color = 'red')

plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

plt.show()



