#
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='jieba')

import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('../model/news12g_bdbk20g_nov90g_dim128.bin', binary=True)
vocabulary = model.wv

keywords = open('../data/keywords.txt').readlines()
keywords = [x.strip('\n') for x in keywords]

output = []
for x in keywords:
	temp_array = np.array(vocabulary[x])
	output.append(temp_array/np.linalg.norm(temp_array))
output = np.array(output)
print(output)
np.save('../model/keyvectors',output)
