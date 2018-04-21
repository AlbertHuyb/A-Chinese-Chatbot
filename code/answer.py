import jieba.posseg as psg
import numpy as np
import gensim

keywords = open('keywords.txt').readlines()
keywords = [x.strip('\n') for x in keywords]
keyvector = np.load('keyvectors.npy')
content = open('test.txt').readlines()
content_vector = np.load('content_vectors.npy')
#model = gensim.models.KeyedVectors.load_word2vec_format('news12g_bdbk20g_nov90g_dim128.bin', binary=True)
#vocabulary = model.wv

question = input("你好！你想了解什么？\n")

aim_tags = []
aim_vector = np.zeros(64)
aim_words = []
temp_tags = psg.cut(question)
for i in temp_tags:
	if i.word in keywords:
		#print(i.word)
		aim_words.append(i.word)
		aim_vector = aim_vector + keyvector[keywords.index(i.word)]
		
	'''if i.word in vocabulary:
			#print(i.word)
			aim_words.append(i.word)
			aim_vector = aim_vector + vocabulary[i.word]'''
length = len(aim_words)

if length != 0:
	aim_vector = aim_vector/length
else:
	print("请换个说法")
	exit(0)

distance = []
for i in range(0,len(content_vector)):
	distance.append([i,np.linalg.norm(aim_vector - content_vector[i]),content[i]])
distance.sort(key = lambda a: a[1])
for i in range(0,5):
	print(distance[i][2])