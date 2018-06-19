import jieba.posseg as psg
import numpy as np

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='jieba')

import gensim

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

keywords = open('../data/keywords.txt',encoding = 'gbk').readlines()
keywords = [x.strip('\n') for x in keywords]
keyvector = np.load('../model/keyvectors.npy')
content = open('../data/test1.txt',encoding = 'gbk').readlines()
content_vector = np.load('../model/content_vectors.npy')
#model = gensim.models.KeyedVectors.load_word2vec_format('news12g_bdbk20g_nov90g_dim128.bin', binary=True)
#vocabulary = model.wv

while(1):
	question = input("你好！你想了解什么？\n")

	aim_tags = []
	aim_vector = np.zeros(64)
	aim_words = []
	words_vectors = [] 
	temp_tags = psg.cut(question)
	for i in temp_tags:
		if i.word in keywords:
			#print(i.word)
			aim_words.append(i.word)
			words_vectors.append(keyvector[keywords.index(i.word)])
			aim_vector = aim_vector + words_vectors[-1]
		
		'''if i.word in vocabulary:
				#print(i.word)
				aim_words.append(i.word)
				aim_vector = aim_vector + vocabulary[i.word]'''
	length = len(aim_words)

	if length != 0:
		aim_vector = aim_vector/length
		#aim_vector = self_attention(vector = aim_vector,word_list = words_vectors,times = 2)
	else:
		print("请换个说法")
		continue

	distance = []
	for i in range(0,len(content_vector)):
		#print(i)
		distance.append([i,np.linalg.norm(aim_vector - content_vector[i]),content[i]])
		#print(distance[-1])
	distance.sort(key = lambda a: a[1])
	print("您好，搜索结果如下：")
	for i in range(0,10):
		print("相似度： %f"%distance[i][1])
		print("内容： %s"%distance[i][2])	
		'''
		if i <= 1:
			print("相似度： %f"%distance[i][1])
			print("内容： %s"%distance[i][2])
		else:
			print(float(distance[i][1])-float(distance[i-1][1]))
			print(float(distance[i-1][1])-float(distance[i-2][1]))
			if((float(distance[i][1])-float(distance[i-1][1])) <= 3*(float(distance[i-1][1])-float(distance[i-2][1]))):
				print("相似度： %f"%distance[i][1])
				print("内容： %s"%distance[i][2])
			else:
				break
		'''