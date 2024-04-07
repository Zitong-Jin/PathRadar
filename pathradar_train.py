import os, math, pickle, argparse
import networkx as nx
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict

class Trainer(object):
    def __init__(self, rounds, quick_mode):
        self.graph = nx.Graph()
        self.relationship = dict()
        self.provider = defaultdict(set)
        self.peer = defaultdict(set)
        self.customer = defaultdict(set)
        self.rounds = rounds
        self.quick_mode = quick_mode
        self.read_topo()
        self.cal_features()

    def read_topo(self):
        with open('asrel_rib.txt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                [asn1, asn2, r] = line.strip().split('|')
                self.graph.add_edge(asn1, asn2)
                if r == '-1':
                    self.relationship[(asn1, asn2)] = 'p2c'
                    self.relationship[(asn2, asn1)] = 'c2p'
                    self.customer[asn1].add(asn2)
                    self.provider[asn2].add(asn1)
                else:
                    self.relationship[(asn1, asn2)] = 'p2p'
                    self.relationship[(asn2, asn1)] = 'p2p'
                    self.peer[asn1].add(asn2)
                    self.peer[asn2].add(asn1)

        self.core_cluster = set()
        with open('core_shell.txt') as f:
            for line in f:
                info = line.strip().split(' ')
                if int(info[1]) > 1000:
                    self.core_cluster.add(info[0])

    def cal_features(self):
        self.linkfreq = defaultdict(int)
        self.linkorder = defaultdict(list)
        self.prev_frequency, self.next_frequency = defaultdict(int), defaultdict(int)
        self.shortest = dict()

        for file in os.listdir('shortest/'):
            with open('shortest/' + file) as f:
                asn = file[:-4]
                self.shortest[(asn, asn)] = 1
                for line in f:
                    path = line.strip().split('|')[::-1]
                    if (path[0], path[-1]) not in self.shortest or len(path) < self.shortest[(path[0], path[-1])]:
                        self.shortest[(path[0], path[-1])] = len(path)

        with open('linkfreq_train.txt') as f:
            for line in f:
                [link, num] = line.strip().split(' ')
                self.linkfreq[tuple(link.split('|'))] = int(num)
            
        with open('linkorder_train.txt') as f:
            for line in f:
                [node, neighbor] = line.strip().split(' ')
                self.linkorder[node] = neighbor.split('|')

        with open('joinfreq_train.txt') as f:
            for line in f:
                [flag, key, num] = line.strip().split(' ')
                if flag == "prev":
                    self.prev_frequency[key] = int(num)
                else:
                    self.next_frequency[key] = int(num)

    def get_paths(self, name):
        sample_paths = defaultdict(set)
        if name == 'train':
            positive_name = 'validation/'
        else:
            positive_name = 'test/'
        negative_name = ['shortest/']
        if self.rounds != 0:
            for i in range(self.rounds):
                negative_name.append('result_round' + str(i) + '/')
        avoid_name = ['rib/']

        def add_sample(path, cate):
            if len(path) < 2:
                return
            slen = self.shortest[(path[0], path[-1])] if (path[0], path[-1]) in self.shortest else len(path)
            if slen <= 3:
                num = '1'
            elif slen <= 5:
                num = '2'
            else:
                num = '3'
            if path[-1] in self.core_cluster:
                sample_paths['core' + num].add((tuple(path), cate))
            if path[-1] not in self.graph.nodes() or self.graph.degree(path[-1]) < 300:
                sample_paths['shell' + num].add((tuple(path), cate))

        unfinished_name = set(os.listdir(negative_name[0]))
        for i in range(1, len(negative_name)):
            unfinished_name &= set(os.listdir(negative_name[i]))
        filenames = os.listdir(positive_name)
        
        cnt = 0
        for file in filenames:
            if file + '.txt' not in unfinished_name:
                continue
            cnt += 1
            if cnt > 1000 and self.quick_mode == 1:
                break
            filestore = set()
            sample_tuple = set()
            with open(positive_name + file) as f:
                for line in f:
                    paths = line.strip().split('&')[0].split(' ')
                    for path in paths:
                        path = path.split('|')[::-1]
                        add_sample(path, 1)
                        filestore.add(tuple(path))
                        sample_tuple.add(path[-1])

            for name in avoid_name:
                if file in os.listdir(name):
                    with open(name + file) as f:
                        for line in f:
                            paths = line.strip().split('&')[0].split(' ')
                            for path in paths:
                                path = path.split('|')[::-1]
                                filestore.add(tuple(path))

            for name in negative_name:
                with open(name + file + '.txt') as f:
                    for line in f:
                        info = line.strip().split(' ')
                        path = tuple(info[0].split('|')[::-1])
                        score = 0.5
                        if len(info) > 1:
                            score = float(info[1])
                        if path[-1] in sample_tuple and path not in filestore and score <= 0.8:
                            add_sample(path, 0)
        return sample_paths

    def get_sample(self, path, cluster):
        att = [len(path), self.shortest[(path[0], path[-1])] if (path[0], path[-1]) in self.shortest else len(path)]
        att.extend([len(self.provider[path[-1]]), len(self.peer[path[-1]]), len(self.customer[path[-1]])])
        att.extend([len(self.provider[path[-2]]), len(self.peer[path[-2]]), len(self.customer[path[-2]])])

        if 'core' in cluster:
            for i in range(len(path), 0, -1):
                if i == 1:
                    att.extend([1, 0])
                else:
                    temp = '|'.join(path[-i:])
                    if temp in self.next_frequency:
                        att.extend([i, self.next_frequency[temp]])
                        break
            att.append(self.linkfreq[(path[-1], path[-2])])
        
        if path[-2] not in self.linkorder[path[-1]]:
            att.append(-1)
        else:
            att.append(self.linkorder[path[-1]].index(path[-2]))
        if path[-1] not in self.linkorder[path[-2]]:
            att.append(-1)
        else:
            att.append(self.linkorder[path[-2]].index(path[-1]))

        for i in range(len(path), 0, -1):
            if i == 1:
                att.extend([1, 0])
            else:
                temp = '|'.join(path[:i])
                if temp in self.prev_frequency:
                    att.extend([i, self.prev_frequency[temp]])
                    break

        rel = []
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) in self.relationship:
                rel.append(self.relationship[(path[i], path[i + 1])])
            else:
                rel.append('p2p')
        att.extend([rel.count('c2p'), rel.count('p2p'), rel.count('p2c')])

        slen, slen1 = (self.shortest[(path[0], path[-1])] if (path[0], path[-1]) in self.shortest else len(path)), (self.shortest[(path[0], path[-2])] if (path[0], path[-2]) in self.shortest else (len(path) - 1))
        att.extend([len(path) - slen, len(path) - 1 - slen1])

        return att

    def get_data(self, key):
        train_atts, train_label = [], []
        for path_pair in self.train_paths[key]:
            (path, ground_truth) = path_pair
            train_label.append(ground_truth)
            train_atts.append(self.get_sample(path, key))
        
        test_atts, test_label = [], []
        for path_pair in self.test_paths[key]:
            (path, ground_truth) = path_pair
            test_label.append(ground_truth)
            test_atts.append(self.get_sample(path, key))
        return train_atts, train_label, test_atts, test_label

    def run(self):
        self.train_paths, self.test_paths = self.get_paths('train'), self.get_paths('test')
        print('data load finished...')
        if not os.path.exists('models/'):
            os.makedirs('models/')
        for cate in ['shell', 'core']:
            for num in ['1', '3', '2']:
                key = cate + num
                train_atts, train_label, test_atts, test_label = self.get_data(key)
                train_atts, train_label = np.array(train_atts), np.array(train_label)
                test_atts, test_label = np.array(test_atts), np.array(test_label)
                print(key, sum(train_label), len(train_label) - sum(train_label))
                if sum(train_label) == 0:
                    continue

                params = {'eval_metric': ['aucpr'], 'scale_pos_weight': (len(train_label) - sum(train_label)) / sum(train_label), 'use_label_encoder': False}#, 'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'
                
                eval_set=[(train_atts, train_label), (test_atts, test_label)]
                model = XGBClassifier(**params)
                model.fit(train_atts, train_label, eval_set=eval_set, verbose=0)

                print(model.feature_importances_)
                pickle.dump(model, open('models/xgboost_' + str(self.rounds) + '_' + str(key) + '.model', 'wb'))
                print('store finished...')
                true_negative, false_positive = 0, 0
                predictions = list(model.predict(test_atts))
                print(len(test_label), len(test_label) - sum(test_label))
                for i in range(len(predictions)):
                    predictions[i] = round(predictions[i])
                    if test_label[i] == 1 and predictions[i] == 0:
                        true_negative += 1
                    if test_label[i] == 0 and predictions[i] == 1:
                        false_positive += 1
                print(key, accuracy_score(test_label, predictions), true_negative, false_positive)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('-r', '--rounds', type=int, required=True)
    parser.add_argument('-q', '--quick_modes', type=int, required=True)
    args = parser.parse_args()
    
    trainer = Trainer(args.rounds, args.quick_modes)
    trainer.run()