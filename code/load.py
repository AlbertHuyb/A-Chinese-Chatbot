#负责找出文本中的关键词，并且存放在Keywords.txt中

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='jieba')

import gensim
import jieba.analyse

model = gensim.models.KeyedVectors.load_word2vec_format('../model/news12g_bdbk20g_nov90g_dim128.bin', binary=True)
vocabulary = model.wv
print("model loaded successfully!")
content = open('../data/test1.txt').read()
stoplist = open('../data/ChineseStoplist.txt').readlines()
stop_list = [x.strip('\n').strip(' ') for x in stoplist]
tags = jieba.analyse.extract_tags(content,300)
output = open('../data/keywords.txt','w')
num = 0
for t in tags:
	if t in stop_list:
		num += 1 
	elif t in vocabulary:
		output.write(t + '\n')
	else:
		print(t)
print(num)


