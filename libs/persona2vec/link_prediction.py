import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from persona2vec.model import Persona2Vec

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class linkPredictionTask(object):
    def __init__(self, G,
                 name,
                 lambd=0.1,
                 num_walks=10,
                 walk_length=40,
                 dimensions=16,
                 window_size=5,
				 workers=4,
				 base_iter=10):
        self.G = G
        self.name = name
        self.lambd = lambd
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.window_size = window_size
		self.workers = workers
		self.base_iter = base_iter

        self.original_edge_set = set(G.edges)
        self.node_list = list(G.nodes)
        self.total_number_of_edges = len(self.original_edge_set)
        self.number_of_test_edges = int(self.total_number_of_edges / 2)
        self.test_edges = []
        self.negative_edges = []

    def train_test_split(self):
        count = 0
        logging.info('Initiate train test set split')
        while count != self.number_of_test_edges:
            edge_list = np.array(self.G.edges())
            candidate_idxs = np.random.choice(
                len(edge_list), self.number_of_test_edges - count, replace=False)
            for source, target in tqdm(edge_list[candidate_idxs]):
                self.G.remove_edge(source, target)
                if nx.is_connected(self.G):
                    count += 1
                    self.test_edges.append((source, target))
                else:
                    self.G.add_edge(source, target, weight=1)

    def generate_negative_edges(self):
        count = 0
        logging.info('Initiate generating negative edges')
        while count != self.number_of_test_edges:
            src, tag = np.random.choice(self.node_list, 2)
            if (src, tag) in self.original_edge_set:
                pass
            else:
                count += 1
                self.negative_edges.append((src, tag))

    def learn_persona2vec_emb(self):
        logging.info('Initiate persona2vec')
        self.model = Persona2Vec(self.G,
                                 lambd=self.lambd,
                                 num_walks=self.num_walks,
                                 walk_length=self.walk_length,
                                 dimensions=self.dimensions,
                                 window_size=self.window_size,
								 workers=self.workers,
								 base_iter=self.base_iter)
        self.model.simulate_walks()
        self.emb = self.model.learn_embedding()

    def calculate_link_prediction_score(self):
        self.link_prediction_score_postive = np.array(
            self.calculate_score(self.test_edges))
        self.link_prediction_score_negative = np.array(
            self.calculate_score(self.negative_edges))

    def calculate_score(self, edge_list):
        score_list = []
        for src, tag in edge_list:
            src_personas = self.model.node_to_persona[src]
            tag_personas = self.model.node_to_persona[tag]
            max_sim = max([np.dot(self.emb[src_persona], self.emb[tag_persona])
                           for src_persona in src_personas for tag_persona in tag_personas])
            score_list.append(max_sim)
        return score_list

    def calculate_ROC_AUC_value(self):
        logging.info('Calcualte ROC_AUC values')
		y_true = np.concatenate([np.ones_like(test_object.link_prediction_score_postive), np.zeros_like(test_object.link_prediction_score_negative)])
		y_score = np.concatenate([test_object.link_prediction_score_postive, test_object.link_prediction_score_negative], axis=0)
		self.ROC_AUC_value = roc_auc_score(y_true, y_score)

    def print_result(self):
        logging.info(self.name)
        logging.info(self.lambd)
        logging.info(self.dimensions)
        logging.info(self.ROC_AUC_value)

    def write_result(self, file_name):
        f = open(file_name, 'a')
        f.write('\t'.join([self.name, str(self.lambd), str(
            self.dimensions), str(self.ROC_AUC_value)]) + '\n')
        f.close()
