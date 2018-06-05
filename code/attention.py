import jieba.posseg as psg
import numpy as np
import gensim
from math import exp

model = gensim.models.KeyedVectors.load_word2vec_format('../model/news12g_bdbk20g_nov90g_dim128.bin', binary=True)
vocabulary = model.wv
list_temp = open('../data/ChineseStoplist.txt').readlines()
stoplist = [i.strip('\n').strip(' ') for i in list_temp]
#load w2vec model
print("model loaded!!")

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

while 1:

	sentence = input()

	words = psg.cut(sentence)

	print("words:")
	print([x.word for x in words])

	vector_list = []

	words = psg.cut(sentence)
	for w in words:
		if w.word in vocabulary:
			if w.word not in stoplist:
				vector_list.append(np.array(vocabulary[w.word]))
				print("len" + str(len(vector_list)))
			else:
				print(w.word)
		else:
			print(w.word)

	vector = np.array(sum(vector_list)/len(vector_list))



	for i in range(0,len(vector_list)):
		print(1/len(vector_list),end = " ")
	print(" ")

	result = self_attention(vector = vector,times = 3)

'''
	temp = []

	for i in range(0,len(vector_list)):
		temp.append(np.dot(vector_list[i],vector)/(np.linalg.norm(vector) * np.linalg.norm(vector_list[i])))

	sumx = sum(temp)
	new =[x/sumx for x in temp ]
	print(new)

	temp = []
	vector = np.zeros(64)
	for i in range(0,len(new)):
		vector = new[i]*np.array(vector_list[i]) + vector

	for i in range(0,len(vector_list)):
		temp.append(np.dot(vector_list[i],vector)/(np.linalg.norm(vector) * np.linalg.norm(vector_list[i])))

	sumx = sum(temp)
	new =[x/sumx for x in temp ]
	print(new)
	#for i in range(0,len())
'''

