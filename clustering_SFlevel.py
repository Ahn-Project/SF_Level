import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans


def load():
    data = np.array(
          [[36, 37, 37, 33, 29, 21],
           [40, 33, 42, 37, 31, 33],
           [40, 42, 42, 37, 38, 29],
           [41, 33, 48, 27, 31, 37],
           [41, 35, 35, 27, 24, 31],
           [41, 46, 42, 36, 27, 33],
           [42, 42, 44, 42, 24, 40],
           [42, 43, 35, 27, 29, 21],
           [42, 37, 40, 43, 43, 37],
           [42, 38, 42, 35, 29, 33],
           [47, 46, 51, 43, 36, 41],
           [48, 45, 54, 48, 42, 36],
           [48, 46, 50, 44, 43, 38],
           [48, 46, 53, 45, 35, 36],
           [48, 50, 53, 48, 42, 29],
           [48, 42, 55, 42, 42, 42],
           [48, 49, 49, 43, 40, 37],
           [48, 50, 52, 42, 38, 40],
           [48, 51, 46, 45, 36, 36],
           [48, 51, 55, 48, 24, 38],
           [45, 44, 48, 36, 37, 40],
           [46, 42, 51, 42, 27, 35],
           [46, 44, 49, 44, 37, 36],
           [46, 46, 52, 33, 40, 38],
           [46, 35, 49, 27, 38, 41],
           [46, 40, 46, 44, 38, 38],
           [46, 40, 49, 40, 40, 38],
           [46, 44, 51, 41, 40, 37],
           [46, 45, 48, 42, 37, 37],
           [47, 37, 52, 45, 31, 42]])

    data = pd.DataFrame(data, columns=['O','S','P','I','E','D'])

    return data

def Clustering():
    data = load()
    arr_data = data.values

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(arr_data)
    predict = kmeans.predict(arr_data)

    data['cluster'] = predict

    data.cluster[data.cluster == 0] = 'Cluster_A'
    data.cluster[data.cluster == 1] = 'Cluster_B'
    data.cluster[data.cluster == 2] = 'Cluster_C'

    return data


def Plot():
    data = Clustering()

    color = ['g','b','r']
    # 다차원 클러스터 시각화
    ax = pd.plotting.parallel_coordinates(data, 'cluster', color=color)
    # label 역순으로 시각화
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc=1)
    plt.show()


if __name__ == '__main__':
    Plot()

