import warnings

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def KNN_compare(data, target, k, split_list, time):
    for split_ratio in split_list:
        micro_list = []
        macro_list = []
        train_list = []
        for i in range(time):
            x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=split_ratio, shuffle=True)
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_true = y_test.reshape(y_pred.shape[0], )
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')
            train_score = clf.score(x_train, y_train)
            macro_list.append(f1_macro)
            micro_list.append(f1_micro)
            train_list.append(train_score)
        print('KNN({}avg, split:{}, k={})train score:{:.4f}, f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
            time, split_ratio, k, sum(train_list) / len(train_list), sum(macro_list) / len(macro_list),
                                  sum(micro_list) / len(micro_list)))


def preprocess_for_knn(embedding, label):
    data = np.round(embedding.astype(float), 7)
    target = np.array(label).reshape(len(label), 1)
    split_list = [0.3, 0.5, 0.7, 0.9]
    warnings.filterwarnings("ignore")
    print("KNN node classification task......")
    KNN_compare(data, target, k=5, split_list=split_list, time=10)
