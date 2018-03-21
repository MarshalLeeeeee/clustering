import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture as mx
import Kmeans
import GMM
import GMM_sklearn

def init(d, n, k, m = None, s = None, kp = None):
	# d is the dimension of each data
	# n is the number of data
	# k is the designed cluster of the generated data
	# cluster_num is the target cluster number in the clustering alrogithm

	if m == None:
		miu = np.random.randn(k,d)
	else:
		miu = m

	if s == None:
		Sigma = np.random.randn(k,d,d)
		for i in range(k):
			S = np.triu(Sigma[i])
			S += S.T - np.diag(S.diagonal())
			Sigma[i] = S
	else:
		Sigma = s

	if kp == None:
		k_num = np.random.random_sample((1,k))
		k_num_sum = np.sum(k_num) * np.ones(k)
		k_partial = np.int_(np.dot(np.true_divide(k_num, k_num_sum), n))
		k_partial = np.squeeze(k_partial, axis=0)
		k_partial[-1] += n - np.sum(k_partial)
		#print(k_partial)
	else:
		k_partial = kp

	start = 0
	data = np.empty([1,d])
	target = np.empty([1,1])
	for i in range(k):
		gen = np.dot(np.random.randn(k_partial[i], d), Sigma[i]) + miu[i]
		target_cluster = np.tile(np.array([i]), (k_partial[i], 1))
		data = np.concatenate((data,gen),axis=0)
		target = np.concatenate((target,target_cluster),axis = 0)
	data = np.concatenate((data[1:,],target[1:,]), axis=1)
	np.random.shuffle(data)
	#print(data)

	return (data, miu, Sigma, k_partial)

def printPara(str, x):
	print(str + ':')
	print(x)


if __name__ == '__main__':
	
	m = [[0,0],[-5,5],[5,-5],[5,5]];
	s = [[[1,-0.2],[-0.2,1]],[[1,0],[0,1]],[[1,0],[0,1]],[[1,-0.8],[-0.8,1]]]
	m1 = [[0,4], [2,-2], [-2,-2]]
	s1 = [[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]]
	m3 = [[0,0],[3,3],[-3,-3]]
	s3 = [[[1,-0.5],[-0.5,1]],[[1,0],[0,1]],[[1,0],[0,1]]]
	m5 = [[0,0],[2,2],[-2,-2],[-2,2],[2,-2]]
	s5 = [[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]]
	kp1000 = [333,333,334]
	data, miu, Sigma, partial = init(2,1000,3,s = s3, m = m3)

	n = data.shape[0]
	d = data.shape[1]-1
	target = data[:,d,np.newaxis]
	data = data[:,:d]
	cm = plt.cm.get_cmap('rainbow')
	cluster = np.array(m3)
	if(d == 2):
		p = np.concatenate((data,target),axis=1)
		plt.figure(figsize=(10,10))
		plt.title('origin')
		plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = 5)
		plt.scatter(cluster[:,0], cluster[:,1],marker = 'o', color = 'r', s = 50)
		#plt.show()
		plt.savefig('./demo/compare/exp1/origin.png')
	#cluster = Kmeans.k_means(data, 4, 1000, Debug = True, debug = 1, slow = True, competitive = True, alpha = 150)
	#print("The cluster is: ")
	#print(cluster)
	#print(data)

	#cluster_miu, cluster_Sigma, cluster_pi = GMM.gmm(data, 4, 100, Debug = True, debug = 1, independent = True)
	#print(cluster_miu)
	#print(cluster_Sigma)
	#print(cluster_pi)
	

	''' save the current data
	np.save('data.npy', data)
	'''

	''' load data from the file
	data = np.load('data.npy')
	'''

	GMM_sklearn.experiment(data,10)

	#gmm = mx.GaussianMixture(n_components = 5)
	#gmm.fit(data)
	#print(gmm.bic(data))
	

