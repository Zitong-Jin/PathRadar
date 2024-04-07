from collections import defaultdict
import os

for cate in ['done_rib', 'rib_train', 'rib_validation', 'rib_data', 'rib_test', 'rib_origintest']:
	name = cate.split('_')[1]
	if not os.path.exists(name):
		os.mkdir(name)

	info = defaultdict(set)
	with open(cate + '.txt') as f:
		for line in f:
			path = line.strip().split('|')
			info[path[-1]].add('|'.join(path))

	for key in info:
		fout = open(name + '/' + key, 'w')
		all_path = set()
		for path in info[key]:
			ases = path.split('|')
			for i in range(len(ases) - 1):
				all_path.add('|'.join(ases[i:]))
		fout.write(' '.join(all_path) + '\n')
		fout.close()