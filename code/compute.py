import gensim
import jieba.posseg as psg
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('news12g_bdbk20g_nov90g_dim128.bin', binary=True)
vocabulary = model.wv

keywords = open('keywords.txt').readlines()
keywords = [x.strip('\n') for x in keywords]
content = open('test.txt').readlines()
vectors = []
num = 0
for c in content:
	num += 1
	word_list = []
	temp_list = psg.cut(c)
	vector = np.zeros(64)
	for i in temp_list:
		print(i.word)
		if i.word in keywords:
			print('yes')
			if i.word in vocabulary:
				word_list.append(i.word)
				temp_array = np.array(vocabulary[i.word])
				vector = vector + temp_array/np.linalg.norm(temp_array)
	temp_length = len(word_list)
	if temp_length != 0:
		vector = vector/temp_length
	print(num)
	print(c)
	print(vector)
	vectors.append(vector)
vectors = np.array(vectors)
np.save("content_vectors",vectors)


