import numpy as np
import matplotlib.pyplot as plt

def distance(data, cluster):
	tmp = data - cluster
	return np.dot(tmp, tmp.T)

def k_means(data, cluster_num, epoch, slow = False, Debug = True, debug = 100):
	# the data is a n*d matrix, where d is the dimension of a single data, and n is the number of the data.

	d = data.shape[1]
	n = data.shape[0]

	if slow:
		cluster = np.random.randn(cluster_num, d)
		print("The initial cluster is:")
		print(cluster)
	else:
		cluster = data[0:cluster_num]

	k = cluster_num
	r = np.zeros((n,k))
	r_ = np.zeros((n,1))

	print(range(2))
	print('d: %d' % d)
	print('n: %d' % n)
	print('k: %d' % k)

	cm = plt.cm.get_cmap('rainbow')
	epsilon = 1e-15
	loss = float('inf')
	flag = 1

	for _ in range(epoch):
		# E: calc the latent variable r
		for i in range(n):
			dist_min = float('inf')
			for j in range(k):
				dist = distance(data[i],cluster[j])
				if (dist < dist_min):
					dist_min = dist
					cluster_index = j
			vec = np.zeros((k))
			vec[cluster_index] = 1
			r[i] = vec
			r_[i][0] = cluster_index
		#print(r)

		# M: modify the center
		for j in range(k):
			s = np.zeros((1,2))
			for i in range(n):
				if(r[i][j]):
					s = np.concatenate((s, data[np.newaxis,i]), axis = 0)
			s = s[1:]
			if(len(s) == 0):
				np.delete(cluster, j, 0)
			else:
				new = np.mean(s, axis = 0)			
				cluster[j] = new

		# loss
		loss_new = 0
		for i in range(n):
			loss_new += distance(data[i], cluster[int(r_[i][0])])
		if(loss - loss_new < epsilon):
			flag = 0
			print('Converge at UID %d' % _)
			return cluster
			exit(0)
		loss = loss_new

		# plot when debug
		if ( Debug and _ % debug == 0):
			if(d == 2):
				p = np.concatenate((data,r_),axis=1)
				plt.figure(figsize=(10,10)) 
				plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = k)
				plt.scatter(cluster[:,0], cluster[:,1],marker = 'o', color = 'r', s = 50)
				plt.show()
			print('Uid: %d, loss: %f' % (_,loss))

	if flag:
		print('Reach the epoch number!')
		return cluster
