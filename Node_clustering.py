import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def Kmeans_compare(x, y, k=4, time=10, return_NMI=False):
    cluster = KMeans(n_clusters=k)
    NMI_list = []
    ARI_list = []
    for i in range(time):
        cluster.fit(x, y)
        y_pred = cluster.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        s2 = adjusted_rand_score(y, y_pred)
        ARI_list.append(s2)
        NMI_list.append(score)
    score = sum(NMI_list) / len(NMI_list)
    s2 = sum(ARI_list) / len(ARI_list)
    print('NMI (10 avg): {:.4f}, ARI (10avg): {:.4f}'.format(score, s2))
    if return_NMI:
        return score


def preprocess_for_kmeans(embedding, label, class_num):
    data = np.round(embedding.astype(float), 7)
    target = np.array(label).reshape(len(label), )
    print("Kmeans node clustering task......")
    Kmeans_compare(data, target, k=class_num)
