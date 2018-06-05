# A-Chinese-Chatbot

这是一个基于长文本和Word2Vec模型的中文聊天机器人

## Word2Vec模型：
借用___dada____分享在新浪微博上的《120G+训练好的word2vec模型（中文词向量）》

## 提取长文本的关键词：
load.py</br>
利用jieba.analyse进行关键词提取，将不在stoplist里而且在word2vec模型里面的词语提取出来作为该长文本的关键词，储存至keywords.txt
（下文中出现的关键词均指这一txt中的关键词）

## 储存关键词对应的向量：
add.py</br>
利用word2vec模型，将keywords.txt里面的词对应的向量储存至keyvectors.npy

## 计算长文本中每一段话对应的语义向量：
compute.py</br>
长文本储存在test.txt
用jieba.posseg对每一句进行分词，将其和关键词做匹配，把每一句中的关键词向量做归一化之后相加取平均，作为这句话的语义向量，并将整篇长文本的语义向量储存至content_vectors.npy

## 应答系统：
answer.py</br>
输入一个问题，提取问题中的关键词，将关键词向量归一化后相加取平均，作为问题的语义向量，计算这一向量和长文本中各语义向量的欧氏距离，取距离较小的5个答案作为输出

$$ \mathbb{A} = { a, b, c, d, e, f} $$
