import gensim
import jieba.analyse

model = gensim.models.KeyedVectors.load_word2vec_format('news12g_bdbk20g_nov90g_dim128.bin', binary=True)
vocabulary = model.wv
print("model loaded successfully!")
content = open('test.txt').read()
stoplist = open('ChineseStoplist.txt').readlines()
tags = jieba.analyse.extract_tags(content,2000)
output = open('keywords.txt','w')
num = 0
for t in tags:
	if t in stoplist:
		num += 1 
	elif t in vocabulary:
		output.write(t + '\n')
	else:
		print(t)
print(num)
