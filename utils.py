import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse
from scipy.sparse import dok_matrix
from tqdm import tqdm


def select_embeddingBylabel(embedding, author_label):
    author = pd.read_csv(author_label, sep='\t', header=None, names=['author_id', 'label', 'author_name'])
    author_id = list(author['author_id'])
    label = list(author['label'])
    z = embedding[author_id]
    return z, label


def build_fusion_mat(graph_mat, dataset):
    print("Graph structure fusion......")
    total_mat = None
    if dataset == 'dblp':
        total_mat = np.zeros((14475, 14475))
    if dataset == 'aminer':
        total_mat = np.zeros((16543, 16543))
    if dataset == 'yelp':
        total_mat = np.zeros((2614, 2614))
    for filename in tqdm(os.listdir(graph_mat)):
        temp = sio.loadmat(graph_mat + filename)["graph_sparse"]
        total_mat += temp
    if dataset == 'dblp':
        sio.savemat('dataset/DBLP/test/Single_DBLP_mat.mat', {"graph_sparse": sparse.csr_matrix(total_mat)})
    if dataset == 'aminer':
        sio.savemat('dataset/AMiner/test/Single_Aminer_mat.mat',
                    {"graph_sparse": sparse.csr_matrix(total_mat)})
    if dataset == 'yelp':
        sio.savemat('dataset/Yelp/test/Single_Yelp_mat.mat', {"graph_sparse": sparse.csr_matrix(total_mat)})


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Dataset():
    def __init__(self):
        self.source_path = str
        self.target_path = str

    def init_path(self, sourcepath, tarpath):
        self.source_path = sourcepath
        self.target_path = tarpath

    def preprocess_dblp(self):
        """
        Preprocess the data set, renumber the originally disordered data, and write the new dataframe into a new file
        """
        print("Renumber the DBLP dataset index...")
        start = time.time()
        author = pd.read_csv('dataset/DBLP/DBLP_four_area/author.txt', sep='\t', header=None,
                             names=['author_id', 'author_name'], dtype=str)
        conf = pd.read_csv('dataset/DBLP/DBLP_four_area/conf.txt', sep='\t', header=None,
                           names=['conf_id', 'conf_name'], dtype=str)
        paper = pd.read_csv('dataset/DBLP/DBLP_four_area/paper.txt', sep='\t', header=None,
                            names=['paper_id', 'paper_name'], dtype=str)
        term = pd.read_csv('dataset/DBLP/DBLP_four_area/term.txt', sep='\t', header=None,
                           names=['term_id', 'term_name'], dtype=str)
        paper_author = pd.read_csv('dataset/DBLP/DBLP_four_area/paper_author.txt', sep='\t', header=None,
                                   names=['paper', 'author'], dtype=str)
        paper_conf = pd.read_csv('dataset/DBLP/DBLP_four_area/paper_conf.txt', sep='\t', header=None,
                                 names=['paper', 'conf'], dtype=str)
        paper_term = pd.read_csv('dataset/DBLP/DBLP_four_area/paper_term.txt', sep='\t', header=None,
                                 names=['paper', 'term'], dtype=str)
        author_label = pd.read_csv('dataset/DBLP/DBLP_four_area/author_label.txt', sep='\t', header=None,
                                   names=['author_id', 'label', 'author_name'], dtype=str)
        paper_label = pd.read_csv('dataset/DBLP/DBLP_four_area/paper_label.txt', sep='\t', header=None,
                                  names=['paper_id', 'label', 'paper_name'], dtype=str)
        conf_label = pd.read_csv('dataset/DBLP/DBLP_four_area/conf_label.txt', sep='\t', header=None,
                                 names=['conf_id', 'label', 'conf_name'], dtype=str)

        author_id = list(author['author_id'])
        conf_id = list(conf['conf_id'])
        paper_id = list(paper['paper_id'])
        term_id = list(term['term_id'])

        for new_author_num, past_author_num in tqdm(enumerate(author_id)):
            author['author_id'].replace(past_author_num, new_author_num, inplace=True)
            paper_author['author'].replace(past_author_num, new_author_num, inplace=True)
            author_label['author_id'].replace(past_author_num, new_author_num, inplace=True)

        for new_paper_num, past_paper_num in tqdm(enumerate(paper_id)):
            paper['paper_id'].replace(past_paper_num, new_paper_num, inplace=True)
            paper_author['paper'].replace(past_paper_num, new_paper_num, inplace=True)
            paper_conf['paper'].replace(past_paper_num, new_paper_num, inplace=True)
            paper_term['paper'].replace(past_paper_num, new_paper_num, inplace=True)
            paper_label['paper_id'].replace(past_paper_num, new_paper_num, inplace=True)

        for new_conf_num, past_conf_num in tqdm(enumerate(conf_id)):
            conf['conf_id'].replace(past_conf_num, new_conf_num, inplace=True)
            conf_label['conf_id'].replace(past_conf_num, new_conf_num, inplace=True)
            paper_conf['conf'].replace(past_conf_num, new_conf_num, inplace=True)

        for new_term_num, past_term_num in tqdm(enumerate(term_id)):
            term['term_id'].replace(past_term_num, new_term_num, inplace=True)
            paper_term['term'].replace(past_term_num, new_term_num, inplace=True)
        paper_author.to_csv('dataset/DBLP/reindex_dblp/paper_author_new.txt', header=None, index=None, sep='\t',
                            encoding='utf-8', float_format='%.0f')
        paper_conf.to_csv('dataset/DBLP/reindex_dblp/paper_conf_new.txt', header=None, index=None, sep='\t',
                          encoding='utf-8', float_format='%.0f')
        paper_term.to_csv('dataset/DBLP/reindex_dblp/paper_term_new.txt', header=None, index=None, sep='\t',
                          encoding='utf-8', float_format='%.0f')
        paper.to_csv('dataset/DBLP/reindex_dblp/paper_new.txt', header=None, index=None, sep='\t', encoding='utf-8',
                     float_format='%.0f')
        author.to_csv('dataset/DBLP/reindex_dblp/author_new.txt', header=None, index=None, sep='\t', encoding='utf-8',
                      float_format='%.0f')
        conf.to_csv('dataset/DBLP/reindex_dblp/conf_new.txt', header=None, index=None, sep='\t', encoding='utf-8',
                    float_format='%.0f')
        term.to_csv('dataset/DBLP/reindex_dblp/term_new.txt', header=None, index=None, sep='\t', encoding='utf-8',
                    float_format='%.0f')
        author_label.to_csv('dataset/DBLP/reindex_dblp/author_label_new.txt', header=None, index=None, sep='\t',
                            encoding='utf-8', float_format='%.0f')
        paper_label.to_csv('dataset/DBLP/reindex_dblp/paper_label_new.txt', header=None, index=None, sep='\t',
                           encoding='utf-8', float_format='%.0f')
        conf_label.to_csv('dataset/DBLP/reindex_dblp/conf_label_new.txt', header=None, index=None, sep='\t',
                          encoding='utf-8', float_format='%.0f')
        end = time.time()
        print("cost:{} mins".format((end - start) / 60))

    def preprocess_aminer(self):
        print("Renumber the AMiner dataset index...")
        start = time.time()
        author = pd.read_csv('dataset/AMiner/subset/id_author.txt', sep='\t', header=None,
                             names=['author_id', 'author_name'], dtype='str')
        conf = pd.read_csv('dataset/AMiner/subset/id_conf.txt', sep='\t', header=None,
                           names=['conf_id', 'conf_name'], dtype='str')
        paper = pd.read_csv('dataset/AMiner/subset/id_paper.txt', sep='\t', header=None,
                            names=['paper_id', 'paper_name'], dtype='str')
        paper_author = pd.read_csv('dataset/AMiner/subset/paper_author.txt', sep='\t', header=None,
                                   names=['paper', 'author'], dtype='str')
        paper_conf = pd.read_csv('dataset/AMiner/subset/paper_conf.txt', sep='\t', header=None,
                                 names=['paper', 'conf'], dtype='str')
        author_label = pd.read_csv('dataset/AMiner/subset/author_label.txt', sep='\t', header=None,
                                   names=['author_id', 'label', 'author_name'], dtype='str')
        author_id = list(author['author_id'])
        conf_id = list(conf['conf_id'])
        paper_id = list(paper['paper_id'])

        for new_author_num, past_author_num in tqdm(enumerate(author_id)):
            author['author_id'].replace(past_author_num, new_author_num, inplace=True)
            paper_author['author'].replace(past_author_num, new_author_num, inplace=True)
            author_label['author_id'].replace(past_author_num, new_author_num, inplace=True)

        for new_paper_num, past_paper_num in tqdm(enumerate(paper_id)):
            paper['paper_id'].replace(past_paper_num, new_paper_num, inplace=True)
            paper_author['paper'].replace(past_paper_num, new_paper_num, inplace=True)
            paper_conf['paper'].replace(past_paper_num, new_paper_num, inplace=True)

        for new_conf_num, past_conf_num in tqdm(enumerate(conf_id)):
            conf['conf_id'].replace(past_conf_num, new_conf_num, inplace=True)
            paper_conf['conf'].replace(past_conf_num, new_conf_num, inplace=True)

        paper_author.to_csv('dataset/AMiner/reindex_aminer/paper_author_new.txt', header=None, index=None,
                            sep='\t',
                            encoding='utf-8')
        paper_conf.to_csv('dataset/AMiner/reindex_aminer/paper_conf_new.txt', header=None, index=None, sep='\t',
                          encoding='utf-8')
        paper.to_csv('dataset/AMiner/reindex_aminer/paper_new.txt', header=None, index=None, sep='\t',
                     encoding='utf-8')
        author.to_csv('dataset/AMiner/reindex_aminer/author_new.txt', header=None, index=None, sep='\t',
                      encoding='utf-8')
        conf.to_csv('dataset/AMiner/reindex_aminer/conf_new.txt', header=None, index=None, sep='\t',
                    encoding='utf-8')
        author_label.to_csv('dataset/AMiner/reindex_aminer/author_label_new.txt', header=None, index=None,
                            sep='\t',
                            encoding='utf-8')
        end = time.time()
        print("cost:{} mins".format((end - start) / 60))

    def preprocess_yelp(self):
        print("Restore the entities in Yelp......")
        start = time.time()
        id_business_file = 'dataset/Yelp/entity/business.txt'
        business_label_file = 'dataset/Yelp/entity/business_label.txt'
        id_reservation_file = 'dataset/Yelp/entity/reservation.txt'
        id_service_file = 'dataset/Yelp/entity/service.txt'
        id_stars_file = 'dataset/Yelp/entity/stars.txt'
        id_user_file = 'dataset/Yelp/entity/users.txt'

        business_category = pd.read_csv('dataset/Yelp/sourcedata/business_category.txt', header=None, sep='\t',
                                        names=['business', 'category'])
        business_reservation = pd.read_csv('dataset/Yelp/sourcedata/business_reservation.txt', header=None, sep='\t',
                                           names=['business', 'reservation'])
        business_service = pd.read_csv('dataset/Yelp/sourcedata/business_service.txt', header=None, sep='\t',
                                       names=['business', 'service'])
        business_stars = pd.read_csv('dataset/Yelp/sourcedata/business_stars.txt', header=None, sep='\t',
                                     names=['business', 'stars'])
        business_user = pd.read_csv('dataset/Yelp/sourcedata/business_user.txt', header=None, sep=' ',
                                    names=['business', 'user', 'count']).drop(['count'], axis=1)
        # Generate id to name mapping file
        b_name = 'business'
        r_name = 'reservation'
        s_name = 'service'
        st_name = 'starts'
        u_name = 'user'
        business_category['business_name'] = business_category['business'].apply(lambda x: b_name + str(x))
        business_reservation['reservation_name'] = business_reservation['reservation'].apply(lambda x: r_name + str(x))
        business_service['service_name'] = business_service['service'].apply(lambda x: s_name + str(x))
        business_stars['stars_name'] = business_stars['stars'].apply(lambda x: st_name + str(x))
        business_user['user_name'] = business_user['user'].apply(lambda x: u_name + str(x))
        business_category.sort_values(by='business').reset_index(drop=True).to_csv(business_label_file, header=None,
                                                                                   index=None, sep='\t',
                                                                                   encoding='utf-8')
        business_category[['business', 'business_name']].sort_values(by='business').reset_index(drop=True).to_csv(
            id_business_file, header=None, index=None, sep='\t', encoding='utf-8')
        business_reservation[['reservation', 'reservation_name']].drop_duplicates().sort_values(
            by='reservation').reset_index(drop=True).to_csv(id_reservation_file, header=None, index=None, sep='\t',
                                                            encoding='utf-8')
        business_service[['service', 'service_name']].drop_duplicates().sort_values(by='service').reset_index(
            drop=True).to_csv(id_service_file, header=None, index=None, sep='\t', encoding='utf-8')
        business_stars[['stars', 'stars_name']].drop_duplicates().sort_values(by='stars').reset_index(
            drop=True).to_csv(id_stars_file, header=None, index=None, sep='\t', encoding='utf-8')
        business_user[['user', 'user_name']].drop_duplicates().sort_values(by='user').reset_index(
            drop=True).to_csv(id_user_file, header=None, index=None, sep='\t', encoding='utf-8')
        end = time.time()
        print("cost:{} mins".format((end - start) / 60))

    def postprocessing(self, output_path, file_name):
        """
        Parallelization and deduplication
        :param output_path: File directory to be deduplicated
        :param file_name: File name under file directory
        """
        file_data = pd.read_csv(open(output_path + file_name, 'r', encoding='utf-8'), sep=' ', header=None)
        re_file_data = file_data.drop_duplicates(keep='first').reset_index(drop=True)
        re_file_data.to_csv(output_path + file_name, header=None, index=None, sep=' ', encoding='utf-8')

    def parallel_process(self, output_path):
        """
        Multi-process method
        :param output_path: Deduplicate file directory
        """
        start = time.time()
        print("Start the meta-path sequence de-duplication process...")
        pool = mp.Pool(mp.cpu_count())
        for file_name in os.listdir(output_path):
            pool.apply_async(self.postprocessing, args=(output_path, file_name))
        pool.close()
        pool.join()
        end = time.time()
        print("cost:{} mins".format((end - start) / 60))

    def build_graph_from_rwfile_mat(self, graph_path, output_path):
        print("Restore matrix from sequence......")
        for file_name in os.listdir(output_path):
            file_name_meta_path = file_name.split('_')[0]
            file_data = pd.read_csv(output_path + file_name, sep=' ', header=None)
            file_data = file_data.loc[:, 0].apply(lambda x: x[1:]).drop_duplicates().reset_index(drop=True).astype(int)
            # Get the number of nodes
            node_num = list(file_data.sort_values(ascending=True))[-1]
            file_read = open(output_path + file_name, 'r', encoding='utf-8', newline='')
            tmp_path = graph_path + file_name_meta_path + '_rw_adjmatrix.mat'
            graph_matrix = dok_matrix((node_num + 1, node_num + 1), dtype=np.int_)
            for line in tqdm(file_read.readlines()):
                line = line.strip().split()
                num_line = len(line)
                for index in range(0, num_line - 1):
                    A = int(line[index][1:])
                    B = int(line[index + 1][1:])
                    graph_matrix[A, B] = 1
                    graph_matrix[B, A] = 1
            graph_matrix = graph_matrix.tocsr()
            sio.savemat(tmp_path, {"graph_sparse": graph_matrix})
