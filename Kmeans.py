import numpy as np
import matplotlib.pyplot as plt

def density(data, ratio = 0.1):
	d = data.shape[1]
	n = data.shape[0]
	d_max = np.amax(data, axis = 0)
	d_min = np.amin(data, axis = 0)
	d_range = d_max - d_min
	compare_n = ratio * d_range

	# smaller value in density map means higher density in the data
	density_map = np.zeros(n) * float('inf')

	for i in range(n):
		cnt = 0.0
		d = 0.0
		for j in range(n):
			if(np.all(np.abs(data[i]-data[j]) < compare_n)):
				cnt += 1
				d += distance(data[j], data[i])
		density_map[i] = d / cnt

	return density_map

def distance(data, cluster):
	tmp = data - cluster
	return np.dot(tmp, tmp.T)

def k_means(data, cluster_num, epoch, slow = False, Debug = True, debug = 100, competitive = False, alpha = 150, beta = 10):
	# the data is a n*d matrix, where d is the dimension of a single data, and n is the number of the data.

	d = data.shape[1]
	n = data.shape[0]
	k = cluster_num
	r = np.zeros((n,k))
	r_ = np.zeros((n,1))
	k_num = np.zeros(k)

	if slow:
		cluster = np.random.randn(cluster_num, d)
		print("The initial cluster is:")
		print(cluster)
	else:
		cluster = data[0:cluster_num]

	if competitive:
		#cluster = np.ones((k,d))
		density_map = density(data, 0.3)
		#print('density_map: ')
		#print(density_map)
		density_center = np.zeros(k)

	print(range(2))
	print('d: %d' % d)
	print('n: %d' % n)
	print('k: %d' % k)

	cm = plt.cm.get_cmap('rainbow')
	epsilon = 1e-80
	loss = float('inf')
	flag = 1
	delete_flag = 0
	delete_index = []
	push = 20

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

		# plot when debug
		if ( Debug and (_ % debug == 0)):
			if(d == 2):
				p = np.concatenate((data,r_),axis=1)
				plt.figure(figsize=(10,10)) 
				plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = k)
				plt.scatter(cluster[:,0], cluster[:,1],marker = 'o', color = 'r', s = 50)
				plt.show()
			print('Uid: %d, loss: %f' % (_,loss))
		
		# find the maximal density in each cluster
		if competitive:
			density_center = np.zeros((k,n))
			for j in range(k):
				#dd = float('inf')
				for i in range(n):
					if(r[i][j]):
						density_center[j][i] = density_map[i]
					else:
						density_center[j][i] = float('inf')
				density_center[j] = np.argsort(density_center[j])

		# loss
		loss_new = 0
		for i in range(n):
			#print('i:%d'%i)
			loss_new += distance(data[i], cluster[int(r_[i][0])])
		for j in range(k):
			if(k_num[j]):
				loss_new += alpha * distance(cluster[j], data[int(density_center[j][1])]) / beta
		if(loss - loss_new < epsilon):
			flag = 0
			print('Converge at UID %d' % _)
			return cluster
			exit(0)
		loss = loss_new

		# M: modify the center
		for j in range(k):
			s = np.zeros((1,d))
			for i in range(n):
				if(r[i][j]):
					s = np.concatenate((s, data[np.newaxis,i]), axis = 0)
			s = s[1:]
			k_num[j] = len(s)
			print('cluster %d: ' % j)
			print(len(s))
			new = np.mean(s, axis = 0)
			if competitive:
				rx = push if push < len(s) else len(s)
				push_vector = np.zeros(d)
				for x in range(rx):
					push_vector += cluster[j] - data[int(density_center[j][x])]
				push_vector = push_vector / rx
				cluster[j] = new + push_vector * alpha / len(s)
			else:		
				cluster[j] = new
		

	if flag:
		print('Reach the epoch number!')
		return cluster


if __name__ == '__main__':
	d = np.array([[1.,2],[4.,0],[3.,9]])
	d_max = np.amax(d, axis = 0)
	d_min = np.amin(d, axis = 0)
	b = np.array([2., 1., float('inf')])
	print(np.amax(d[:,0]))
	print(np.amax(d[:,1]))
	print(np.amin(d[:,0]))
	print(np.amin(d[:,1]))
	print(np.amax(d, axis = 0))
	print(np.amin(d, axis = 0))
	print(np.abs(d[0] - d[1]))
	print((d_max - d_min) * 0.1)
	print(np.abs(d[0] - d[1]) < (d_max - d_min) * 0.1)
	print(np.argsort(b))
	

