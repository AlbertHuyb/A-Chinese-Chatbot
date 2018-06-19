import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import jieba.posseg as psg
import numpy as np
from math import exp

def self_attention(vector,word_list = None,times = 2):
	for time in range(0,times):
		temp = []
		for i in range(0,len(word_list)):
			temp.append(np.dot(word_list[i],vector)/(np.linalg.norm(vector) * np.linalg.norm(word_list[i])))

		sumx = sum(temp)
		new =[x/sumx for x in temp ]

		vector = np.zeros(64)
		for i in range(0,len(new)):
			vector = new[i]*np.array(word_list[i]) + vector
		print(new)

	return vector

model = gensim.models.KeyedVectors.load_word2vec_format('../model/news12g_bdbk20g_nov90g_dim128.bin', binary=True)
vocabulary = model.wv

keywords = open('../data/keywords.txt').readlines()
keywords = [x.strip('\n') for x in keywords]
content = open('../data/test1.txt').readlines()
vectors = []
num = 0
for c in content:	
	num += 1
	word_list = []
	word_vectors = []
	temp_list = psg.cut(c)
	vector = np.zeros(64)
	for i in temp_list:
		#print(i.word)
		if i.word in keywords:
			#print('yes')
			if i.word in vocabulary:
				word_list.append(i.word)
				#print(keywords.index(i.word))
				temp_array = np.array(vocabulary[i.word])
				#word_vectors.append([temp_array/np.linalg.norm(temp_array),exp(-keywords.index(i.word))])
				word_vectors.append([temp_array/np.linalg.norm(temp_array)])
				vector = vector + np.array(word_vectors[-1])
				#vector = vector + temp_array/np.linalg.norm(temp_array)
	temp_length = len(word_list)
	if temp_length != 0:
		#print(temp_length)

		#total_weight = sum(y[1] for y in word_vectors)
		#vector = sum(x[0]*x[1]/total_weight for x in word_vectors)

		#print([y[1] for y in word_vectors])
		#vector = sum(x[0] for x in word_vectors)
		vector = vector/temp_length
		#vector_list = [vocabulary[x] for x in word_list]

		#vector = self_attention(vector = vector[0],word_list = word_vectors,times = 1)
	
		#print(num)
		#print(c)
		#print(len(c))
		#print(vector)
		vectors.append(vector)
	else:
		vectors.append(vector)
		print("abort!!!!!!!")
vectors = np.array(vectors)
np.save("../model/content_vectors",vectors)


