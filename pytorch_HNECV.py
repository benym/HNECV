import argparse
import os
import sys
import warnings

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from GMM import initMiu_by_GMM
from Node_classifier import preprocess_for_knn
from Node_clustering import preprocess_for_kmeans
from TSNE import preprocess_for_tsne
from base_AE import baseAE
from utils import select_embeddingBylabel


class HNECV(nn.Module):
    def __init__(self, input, h1_dim, h2_dim, z_dim, class_num):
        super(HNECV, self).__init__()
        self.enc_1 = nn.Linear(input, h1_dim)
        self.enc_2 = nn.Linear(h1_dim, h2_dim)
        self.modelEx = nn.Linear(h2_dim, z_dim)
        self.modelEn = nn.Linear(h2_dim, z_dim)
        self.modelHe = nn.Linear(h2_dim, z_dim)
        self.dec_1 = nn.Linear(z_dim, h2_dim)
        self.dec_2 = nn.Linear(h2_dim, h1_dim)
        self.output = nn.Linear(h1_dim, input)
        self.priorEx = Parameter(torch.Tensor(class_num, z_dim))
        torch.nn.init.xavier_normal_(self.priorEx.data)

    def encode(self, x):
        h1 = F.relu(self.enc_1(x))
        h2 = F.relu(self.enc_2(h1))
        return self.modelEx(h2), self.modelEn(h2), self.modelHe(h2)

    def reparameterize(self, Ex, En, He):
        noise1 = torch.randn_like(En)
        noise2 = torch.randn_like(noise1)
        z = Ex + noise1 * En + noise1 * noise2 * He
        Enn = noise1 * He + En
        return z, Enn

    def decode(self, z):
        h3 = F.relu(self.dec_1(z))
        h4 = F.relu(self.dec_2(h3))
        return self.output(h4)

    def forward(self, x):
        Ex, En, He = self.encode(x)
        z, Enn = self.reparameterize(Ex, En, He)
        x_reconst = self.decode(z)
        eps = 1e-8
        one_term = En.pow(2)
        two_term = Ex.pow(2)
        three_term = torch.log(eps + En.pow(2))
        t = 1.0
        q = torch.exp(-(t + torch.sum(torch.pow(z.unsqueeze(1) - self.priorEx.unsqueeze(0), 2), 2)) / (
                t + 2 * torch.sum(torch.pow(Enn.unsqueeze(1), 2), 2)))
        q = (q.t() / torch.sum(q, 1)).t()
        return x_reconst, one_term, two_term, three_term, q


def assist_distribution(q):
    weight = torch.pow(q, 2) / q.sum()
    return (weight.t() / weight.sum(1)).t()


def Reconstruct_loss(X_recon, X, beta):
    B = X * (beta - 1) + 1
    reconstruction_loss = torch.sum(torch.pow((X_recon - X) * B, 2))
    return reconstruction_loss


def pre_train(base_AEmodel, VAEmodel, data_loader, pretrain_epoch, class_num, data_loader2, pretrain_path, beta,
              learning_rate, device):
    target = 0.001
    if not os.path.exists(pretrain_path):
        preopti = torch.optim.Adam(base_AEmodel.parameters(), lr=learning_rate)
        print("Pretraining......")
        for _ in tqdm(range(pretrain_epoch)):
            for _, x in enumerate(data_loader):
                x = x.to(device)
                x_reconst = base_AEmodel(x)
                ploss = Reconstruct_loss(x_reconst, x, beta)
                preopti.zero_grad()
                ploss.backward()
                preopti.step()

        with torch.no_grad():
            z = None
            for data in data_loader2:
                data = data.to(device)
                z, _ = base_AEmodel.encode(data)
            z = z.cpu().numpy()
        torch.save(base_AEmodel.state_dict(), pretrain_path)
        VAEmodel.load_state_dict(torch.load(pretrain_path, map_location=torch.device('cpu')))
        Ex, _, _ = initMiu_by_GMM(z, class_num, target)
        VAEmodel.priorEx.data = torch.tensor(Ex).to(device)
    else:
        base_AEmodel.load_state_dict(torch.load(pretrain_path, map_location=torch.device('cpu')))
        with torch.no_grad():
            z = None
            for data in data_loader2:
                data = data.to(device)
                z, _ = base_AEmodel.encode(data)
            z = z.cpu().numpy()
        VAEmodel.load_state_dict(torch.load(pretrain_path, map_location=torch.device('cpu')))
        Ex, _, _ = initMiu_by_GMM(z, class_num, target)
        VAEmodel.priorEx.data = torch.tensor(Ex).to(device)
    return VAEmodel


def train_VAE(N, learning_rate, n_epochs, pretrain_epochs, batch_size, class_num, loss_path, model_path,
              pretrain_path, beta, data_loader, data_loader2, device):
    loss_file = open(loss_path, 'w', encoding='utf-8', newline='')
    model_vae = HNECV(N[1], 500, 2000, 128, class_num).to(device)
    base_ae = baseAE(N[1], 500, 2000, 128, class_num).to(device)
    print(model_vae)
    model_vae = pre_train(base_ae, model_vae, data_loader, pretrain_epochs, class_num, data_loader2, pretrain_path,
                          beta, learning_rate, device)
    optimizer = torch.optim.Adam(model_vae.parameters(), lr=learning_rate)
    min_loss = sys.maxsize
    for epoch in range(n_epochs):
        train_all_loss, train_recon_loss, train_kl_loss, train_kl_self = 0, 0, 0, 0
        n_bathches = N[0] // batch_size + 1
        for i, x in enumerate(data_loader):
            x = x.to(device)
            x_reconst, one_term, two_term, three_term, q = model_vae(x)
            reconst_loss = Reconstruct_loss(x_reconst, x, beta)
            p = assist_distribution(q)
            kl_div = -0.5 * torch.sum(1 + three_term - two_term - one_term)
            kl_self = F.kl_div(q.log(), p, reduction='batchmean')
            loss = reconst_loss + kl_div + kl_self
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_all_loss += loss.item()
            train_recon_loss += reconst_loss.item()
            train_kl_loss += kl_div.item()
            train_kl_self += kl_self.item()
            if (i + 1) % 1 == 0:
                print(
                    "Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.8f}, KL Loss: {:.8f}, KL Self Loss: {:.8f}"
                        .format(epoch + 1, n_epochs, i + 1, len(data_loader), reconst_loss.item(), kl_div.item(),
                                kl_self.item()))
        train_all_loss /= n_bathches
        train_recon_loss /= n_bathches
        train_kl_loss /= n_bathches
        train_kl_self /= n_bathches

        if train_all_loss < min_loss:
            min_loss = train_all_loss
            print("save model...")
            torch.save(model_vae.state_dict(), model_path)

        print(
            "\rEpochs:{}\tTrain Total loss:{}\tReconstruction loss:{}\tKL loss:{}\tKL Self Loss:{}".format(
                epoch, train_all_loss,
                train_recon_loss,
                train_kl_loss,
                train_kl_self
            ))
        print(
            "\rEpochs:{}\tTrain Total loss:{}\tReconstruction loss:{}\tKL loss:{}\tKL Self Loss:{}".format(
                epoch, train_all_loss,
                train_recon_loss,
                train_kl_loss,
                train_kl_self
            ),
            file=loss_file)
    loss_file.flush()

    with torch.no_grad():
        model_vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        z = None
        for data in data_loader2:
            data = data.to(device)
            Ex, En, He = model_vae.encode(data)
            z, Enn = model_vae.reparameterize(Ex, En, He)
        z = z.cpu().numpy()
        return z


def getdataloader(input_path, batch_size, device):
    adj_matrix = sio.loadmat(input_path)["graph_sparse"]
    N = adj_matrix.shape
    print("Number of nodes:{}".format(N[0]))
    adj_matrix = adj_matrix.toarray()
    dataset = torch.Tensor(adj_matrix).to(device)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    data_loader2 = torch.utils.data.DataLoader(dataset=dataset, batch_size=N[0], shuffle=False)
    return N, data_loader, data_loader2


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--beta', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--pretrain_path', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--embedding_path', type=str)
    parser.add_argument('--loss_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--isValidate', type=bool, default=True)
    parser.add_argument('--savembedding', type=bool, default=True)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.dataset == 'dblp':
        args.M = 4
        args.lr = 0.0001
        args.batch = 128
        args.beta = 2
        args.n_epochs = 100
        args.pretrain_epochs = 30
        args.pretrain_path = "./pretrain_ae.pkl"
        args.input_path = "dataset/DBLP/SingleDBLP.mat"
        args.embedding_path = "modelResult/DBLP/dblp_embedding.txt"
        args.loss_path = "loss.txt"
        args.model_path = "modelResult/DBLP/dblp.pkl"
        args.label_path = "dataset/DBLP/reindex_dblp/author_label_new.txt"
        args.isValidate = True
        args.savembedding = True

    if args.dataset == 'aminer':
        args.M = 8
        args.lr = 0.0001
        args.batch = 128
        args.beta = 5
        args.n_epochs = 100
        args.pretrain_epochs = 30
        args.pretrain_path = "./pretrain_ae.pkl"
        args.input_path = "dataset/AMiner/SingleAminer.mat"
        args.embedding_path = "modelResult/AMiner/aminer_embedding.txt"
        args.loss_path = "loss.txt"
        args.model_path = "modelResult/AMiner/aminer.pkl"
        args.label_path = "dataset/AMiner/reindex_aminer/author_label_new.txt"
        args.isValidate = True
        args.savembedding = True

    if args.dataset == 'yelp':
        args.M = 3
        args.lr = 0.0003
        args.batch = 32
        args.beta = 15
        args.n_epochs = 100
        args.pretrain_epochs = 50
        args.pretrain_path = "./pretrain_ae.pkl"
        args.input_path = "dataset/Yelp/SingleYelp.mat"
        args.embedding_path = "modelResult/Yelp/yelp_embedding.txt"
        args.loss_path = "loss.txt"
        args.model_path = "modelResult/Yelp/yelp.pkl"
        args.label_path = "dataset/Yelp/entity/business_label.txt"
        args.isValidate = True
        args.savembedding = True

    N, data_loader, data_loader2 = getdataloader(args.input_path, args.batch, device)
    print(args)
    z = train_VAE(N, args.lr, args.n_epochs, args.pretrain_epochs, args.batch, args.M,
                  args.loss_path, args.model_path,
                  args.pretrain_path, args.beta, data_loader, data_loader2, device)
    if args.isValidate == True:
        selectedEmbedding, selectedlabel = select_embeddingBylabel(z, args.label_path)
        if args.savembedding == True:
            np.savetxt(args.embedding_path, selectedEmbedding, delimiter=',', fmt='%.07f')
        preprocess_for_knn(selectedEmbedding, selectedlabel)
        preprocess_for_kmeans(selectedEmbedding, selectedlabel, args.M)
        if args.dataset == 'dblp':
            preprocess_for_tsne(selectedEmbedding, selectedlabel)
    else:
        np.savetxt(args.embedding_path, z, delimiter=',', fmt='%.07f')
