import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

#keyvector = np.load('../model/keyvectors.npy')
#print("load keyword!")
con = open('../data/test.txt').readlines()
content_vector = np.load('../model/content_vectors.npy')
print("load content!")
#key = TSNE(random_state = 20150101).fit_transform(np.vstack(keyvector))
content = TSNE(random_state = 20150101).fit_transform(np.vstack(content_vector))
print("transformed!")
for i in range(0,len(content)):
	if content[i][0] < -3 and content[i][1] < -4:
		print(con[i])
plt.scatter(content[:,0],content[:,1])
plt.show()
