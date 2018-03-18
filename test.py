import numpy as np
import matplotlib.pyplot as plt
import Kmeans
import GMM

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
	for i in range(k):
		gen = np.dot(np.random.randn(k_partial[i], d), Sigma[i]) + miu[i]
		#print(gen)
		data = np.concatenate((data,gen),axis=0)
		#print(data)
	data = data[1:]
	np.random.shuffle(data)
	#print(data)

	return (data, miu, Sigma, k_partial)

def printPara(miu, Sigma, partial):
	print('miu:')
	print(miu)
	print('Sigma:')
	print(Sigma)
	print('partial:')
	print(partial)


if __name__ == '__main__':
	
	m = [[0,0],[-5,5],[5,-5],[5,5]];
	s = [[[1,-0.2],[-0.2,1]],[[1,0],[0,1]],[[1,0],[0,1]],[[1,-0.8],[-0.8,1]]]
	data, miu, Sigma, partial = init(2,1000,4,s = s[:4], m = m)

	#cluster = Kmeans.k_means(data, 4, 1000, Debug = True, debug = 1, slow = True)
	#print("The cluster is: ")
	#print(cluster)
	#print(data)
	#np.seterr(all='raise')
	cluster_miu, cluster_Sigma, cluster_pi = GMM.gmm(data, 4, 100, Debug = True, debug = 1)
	print(cluster_miu)
	print(cluster_Sigma)
	print(cluster_pi)


	''' save the current data
	np.save('data.npy', data)
	'''

	''' load data from the file
	data = np.load('data.npy')
	'''
	#plt.figure(figsize=(10,10)) 
	#plt.scatter(data[:,0],data[:,1],marker = '+',color = 'r')
	#plt.show()

	

