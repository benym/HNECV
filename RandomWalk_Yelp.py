
import csv
import multiprocessing as mp
import os
import random
import time

import numpy as np
from tqdm import tqdm



class GeneratorMetaPath_by_randomwalk_yelp:
    def __init__(self):
        self.business_reservation = dict()
        self.reservation_business = dict()
        self.business_service = dict()
        self.service_business = dict()
        self.business_stars = dict()
        self.stars_business = dict()
        self.business_user = dict()
        self.user_business = dict()

    def build_adj_dict(self, dirpath, filename, entityA_entityB_dict, entityB_entityA_dict):
        """
        Construct a dict between entityA_id and entityB_id (bi-directional build)
        :param dirpath: Initial data set path
        :param filename: The file name of the dict whose bidirectional id needs to be constructed
        :param entityA_entityB_dict: Corresponding to the dict of A_B in self
        :param entityB_entityA_dict: Corresponding to the dict of B_A in self
        """
        with open(dirpath + filename) as file:
            for line in file:
                line_words = line.strip().split("\t")
                A, B = line_words[0], line_words[1]
                if A not in entityA_entityB_dict:
                    entityA_entityB_dict[A] = []
                entityA_entityB_dict[A].append(B)
                if B not in entityB_entityA_dict:
                    entityB_entityA_dict[B] = []
                entityB_entityA_dict[B].append(A)

    def parallel_callback(self, file_data):
        """
        Parallel callback, sub-process data centralization, collective writing
        :param file_data: dict type, key is the file path, value corresponds to the sequence of files to be written
        """
        output_dir = list(file_data.keys())[0]
        output_file = csv.writer(open(output_dir, 'a', encoding='utf-8', newline=""), delimiter=' ')
        for i in file_data[output_dir]:
            output_file.writerow(i)

    def parallel_random_walk(self, method, outfilename, numwalks, walklength):
        """
        Parallelized random walk
        :param method: Random walk method corresponding to metapath
        :param outfilename: output file name
        :param numwalks: Number of rows generated by a single start node
        :param walklength: walk length
        """
        start = time.time()
        global file_data
        file_data = dict()
        file_data[outfilename] = list()
        if os.path.exists(outfilename):
            with open(outfilename, 'r+') as file:
                file.truncate()
        paper = list(self.business_service.keys())
        paper_subset = np.array_split(paper, 4)
        pool = mp.Pool(mp.cpu_count())
        for s_paper in paper_subset:
            pool.apply_async(method, args=(s_paper, outfilename, file_data, numwalks, walklength),
                             callback=self.parallel_callback)
        pool.close()
        pool.join()
        end = time.time()
        print("cost:{} mins".format((end - start) / 60))

    def random_walk_by_BSB(self, sub_movieSet, outfilename, file_data, numwalks, walklength):
        """
        Random walk with BSB as metapath
        :param sub_movieSet: Author subset sent by a single process
        :param outfilename: output file name
        :param file_data: Global write file dict
        :param numwalks: Number of rows generated by a single start node
        :param walklength: walk length
        :return: Generated metapath sequence data
        """
        for movie in tqdm(sub_movieSet):
            for i in range(numwalks):
                single_node_seq = []
                single_node_seq.append("B" + movie)
                for j in range(walklength):
                    previous_movie = single_node_seq[-1][1:]
                    actors = self.business_service[previous_movie]
                    next_actor = random.choice(actors)
                    movies = self.service_business[next_actor]
                    next_movie = random.choice(movies)
                    single_node_seq.append("B" + str(next_movie))
                file_data[outfilename].append(single_node_seq)
        return file_data

    def random_walk_by_BStB(self, sub_movieSet, outfilename, file_data, numwalks, walklength):
        """
        Random walk with BStB as metapath
        :param sub_movieSet: Author subset sent by a single process
        :param outfilename: output file name
        :param file_data: Global write file dict
        :param numwalks: Number of rows generated by a single start node
        :param walklength: walk length
        :return: Generated metapath sequence data
        """
        for movie in tqdm(sub_movieSet):
            for i in range(numwalks):
                single_node_seq = []
                single_node_seq.append("B" + movie)
                for j in range(walklength):
                    previous_movie = single_node_seq[-1][1:]
                    years = self.business_stars[previous_movie]
                    next_year = random.choice(years)
                    movies = self.stars_business[next_year]
                    next_movie = random.choice(movies)
                    single_node_seq.append("B" + str(next_movie))
                file_data[outfilename].append(single_node_seq)
        return file_data

    def random_walk_by_BUB(self, sub_movieSet, outfilename, file_data, numwalks, walklength):
        """
        Random walk with BUB as metapath
        :param sub_movieSet: Author subset sent by a single process
        :param outfilename: output file name
        :param file_data: Global write file dict
        :param numwalks: Number of rows generated by a single start node
        :param walklength: walk length
        :return: Generated metapath sequence data
        """
        for movie in tqdm(sub_movieSet):
            for i in range(numwalks):
                single_node_seq = []
                single_node_seq.append("B" + movie)
                for j in range(walklength):
                    previous_movie = single_node_seq[-1][1:]
                    years = self.business_user[previous_movie]
                    next_year = random.choice(years)
                    movies = self.user_business[next_year]
                    next_movie = random.choice(movies)
                    single_node_seq.append("B" + str(next_movie))
                file_data[outfilename].append(single_node_seq)
        return file_data
