from collections import defaultdict
import copy, os, argparse
from multiprocessing import Pool

class Routingtree(object):
    def __init__(self, max_limit):
        self.provider = defaultdict(set)
        self.peer = defaultdict(set)
        self.customer = defaultdict(set)
        self.max_limit = max_limit
        self.read_topo()
    
    def read_topo(self):
        with open('asrel_rib.txt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                [asn1, asn2, rel] = line.strip().split('|')
                if rel == '0' or rel == '1':
                    self.peer[asn1].add(asn2)
                    self.peer[asn2].add(asn1)
                elif rel == '-1':
                    self.customer[asn1].add(asn2)
                    self.provider[asn2].add(asn1)

    def simulate(self, asn):
        self.AS_path = defaultdict(list)
        self.worst = defaultdict(list)
        self.path_flag = dict()
        self.AS_path[asn] = [[asn]]
        self.worst[asn] = [asn]
        self.path_flag[tuple([asn])] = 0
        self.stream(asn)
        
        destination = set()
        with open(self.origin_dir + asn) as f:
            for line in f:
                paths = line.strip().split('&')[0]
                for path in paths.split(' '):
                    destination |= set(path.split('|'))
        fout = open(self.dst_dir + asn + '.txt', 'w')
        for des in destination:
            if asn != des:
                for path in self.AS_path[des]:
                    fout.write('|'.join(path[::-1]) + '\n')
        fout.close()

    def condition(self, current, neighbor):
        if neighbor in current or len(current) >= 9:
            return False
        valleyfree = False
        if self.path_flag[current] == 0:
            valleyfree = True
        elif self.path_flag[current] != 0 and neighbor in self.customer[current[-1]]:
            valleyfree = True
        return valleyfree

    def smaller(self, link1, link2):
        (asn1, asn2) = link1
        (asn3, asn4) = link2
        if asn1 in self.customer[asn2] and (asn3 in self.peer[asn4] or asn3 in self.provider[asn4]):
            return True
        elif asn1 in self.peer[asn2] and asn3 in self.provider[asn4]:
            return True
        return False

    def stream(self, asn):
        active, candicate = [tuple([asn])], set()
        relationship_list = [self.provider, self.peer, self.customer]
        
        while len(active) > 0:
            for i in range(3):
                relationship = relationship_list[i]
                for current in active:
                    for neighbor in relationship[current[-1]]:
                        if self.condition(current, neighbor):
                            right_path = list(current) + [neighbor]
                            if len(self.AS_path[neighbor]) < self.max_limit:
                                self.AS_path[neighbor].append(right_path)
                                self.path_flag[tuple(right_path)] = i
                                candicate.add(tuple(right_path))
                            else:
                                for i in range(len(self.AS_path[neighbor])):
                                    if len(self.AS_path[neighbor][i]) > len(right_path):
                                        self.AS_path[neighbor].pop()
                                        self.AS_path[neighbor].insert(i, right_path)
                                        self.path_flag[tuple(right_path)] = i
                                        candicate.add(tuple(right_path))
                                        break
                            if len(self.worst[neighbor]) == 0 or self.smaller((current[-1], neighbor), (self.worst[neighbor][-2], self.worst[neighbor][-1])):
                                self.worst[neighbor] = copy.deepcopy(right_path)
                                self.path_flag[tuple(right_path)] = i
                                candicate.add(tuple(right_path))

            active = list(candicate)
            active.sort(key = lambda x: int(x[-1]))
            candicate = set()

    def run(self):
        self.origin_dir = 'rib/'
        self.dst_dir = 'shortest/'
        if not os.path.exists(self.dst_dir):
            os.mkdir(self.dst_dir)
        pool = Pool(50)
        pool.map(self.simulate, os.listdir(self.origin_dir))
        pool.close()
        pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='false samples')
    parser.add_argument('-m', '--max_limit', type=int, required=True)
    args = parser.parse_args()
    
    rt = Routingtree(args.max_limit)
    rt.run()
