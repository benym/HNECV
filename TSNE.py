
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

def plot_embedding(data, label, title):
    plt.figure()
    for i in range(data.shape[0]):
        if label[i] == 0:
            color = '#32CD32'
        if label[i] == 1:
            color = '#EEEE00'
        if label[i] == 2:
            color = '#00008B'
        if label[i] == 3:
            color = '#551A8B'
        plt.scatter(data[i, 0], data[i, 1],s=4, c=color)
    plt.scatter(data[:, 0], data[:, 1], s=4, c=label)
    plt.title(title)
    fonts = {'family': 'Times New Roman', 'style': 'italic', 'size': 15}
    plt.xlabel("X", fonts)
    plt.ylabel("Y", fonts)
    plt.savefig('T-SNE.png',dpi=300)
    plt.show()


def preprocess_for_tsne(embedding, label):
    data = np.round(embedding.astype(float), 7)
    label = np.array(label)
    print('Computing T-SNE......')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    plot_embedding(result, label, 'T-SNE task')
