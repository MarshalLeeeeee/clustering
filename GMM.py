import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import traceback
import sys
import Kmeans

def gaussian(data, miu, Sigma, delta = 0):
	# data is d-dimensional vector
	# miu ia d-dimensional vector
	# con is d*d matrix
	d = len(data)
	data_reg = data - miu
	S_det = np.linalg.det(Sigma + np.eye(d)*0.01)
	S_inv = np.linalg.inv(Sigma + np.eye(d)*0.01)
	prob = 1.0/(np.power(2*np.pi,1.0*d/2) + 1e-10)/(np.sqrt(np.abs(S_det)) + 1e-10)*np.exp(delta-1.0/2*np.dot(np.dot(data_reg, S_inv), data_reg.T))

	#print(delta-1.0/2*np.dot(np.dot(data_reg, S_inv), data_reg.T))
	'''
	try:
		prob = 1.0/(np.power(2*np.pi,1.0*d/2) + 1e-10)/(np.sqrt(np.abs(S_det)) + 1e-10)*np.exp(delta-1.0/2*np.dot(np.dot(data_reg, S_inv), data_reg.T))
	except:
		traceback.print_exc()
		print(delta-1.0/2*np.dot(np.dot(data_reg, S_inv), data_reg.T))
	'''
	return prob


def gmm(data, cluster_num, epoch, Debug = True, debug = 100):

	#matplotlib.use('Agg') 

	print('data:')
	print(data)
	print('------------------')

	d = data.shape[1]
	n = data.shape[0]
	k = cluster_num
	r = np.zeros((n,k))
	r_ = np.zeros((n,1))
	p = np.zeros((n,k))
	p_margin = np.zeros(n)
	member = np.zeros(k)
	member_all = np.ones(k) * n
	epsilon = 1e-15
	ll = -float('inf')
	flag = 1
	cm = plt.cm.get_cmap('rainbow')

	# initialize the cluster
	#cluster_miu = Kmeans.k_means(data, cluster_num, 10, Debug = False, debug = 1, slow = True)
	cluster_miu = np.random.randn(k,d)
	cluster_Sigma = np.random.randn(k,d,d)
	for i in range(k):
		'''
		S = np.triu(cluster_Sigma[i])
		S += S.T - np.diag(S.diagonal())
		cluster_Sigma[i] = S
		'''
		cluster_Sigma[i] = np.eye(d) *  5
	cluster_pi = 1.0 / k * np.ones(k)
	#print('initial cluster_pi: ')
	#print(cluster_pi)

	print('initial miu:')
	print(cluster_miu)
	print('initial Sigma:')
	print(cluster_Sigma)
	print('cluster pi:')
	print(cluster_pi)
	print('--------------------')


	for _ in range(epoch):
		# calculate the probability
		#print(p.shape)
		for i in range(n):
			#p_margin[i] = 0.0
			for j in range(k):
				try:
					p[i][j] = cluster_pi[j] * gaussian(data[i], cluster_miu[j], cluster_Sigma[j])
				except:
					print('Gaussian calc when i: %d, j: %d' % (i,j))
					traceback.print_exc()
					sys.exit(1)
			
			cnt = 0
			while(np.sum(p[i]) == 0):
				cnt += 1
				for j in range(k):
					try:
						p[i][j] = cluster_pi[j] * gaussian(data[i], cluster_miu[j], cluster_Sigma[j], np.power(2,cnt))
					except:
						print('Gaussian calc when i: %d, j: %d, cnt %d' % (i,j,cnt))
						traceback.print_exc()
						sys.exit(1)
			cnt2 = 0
			while(np.sum(p[i]) == float('inf')):
				cnt2 += 1
				for j in range(k):
					try:
						p[i][j] = cluster_pi[j] * gaussian(data[i], cluster_miu[j], cluster_Sigma[j], np.power(2,cnt)-np.power(2,cnt2))
					except:
						print('Gaussian calc when i: %d, j: %d, cnt %d' % (i,j,cnt))
						traceback.print_exc()
						sys.exit(1)
		#print('*******************')
		print('UID %d' % _)
		print('p:')
		print(p)
		p_margin = np.sum(p, axis = 1)
		print('p_margin: ')
		print(p_margin)

		# E: calculate the belonging
		for i in range(n):
			m_prob = 0
			for j in range(k):
				r[i][j] = (p[i][j] * 1e+1) / (p_margin[i] * 1e+1)
				if(r[i][j] > m_prob):
					m_prob = r[i][j]
					cluster_index = j
			r_[i] = cluster_index
		print('r:')
		print(r)

		# calculate the members in each cluster
		member = np.sum(r, axis = 0)

		# M: modify the cluster, the miu and Sigma
		cluster_miu = np.true_divide(np.dot(r.T, data), np.tile(member[np.newaxis, :].T, (1,d)))
		for j in range(k):
			cluster_Sigma[j] = np.zeros((d,d))
			for i in range(n):
				data_reg = np.add(data[i], -1*cluster_miu[j])[np.newaxis,]
				S = np.dot(data_reg.T, data_reg)
				cluster_Sigma[j] = np.add(cluster_Sigma[j], r[i][j] * S)
			cluster_Sigma[j] = cluster_Sigma[j] * (1.0 / member[j])
		cluster_pi = np.true_divide(member, member_all)
		print('new_cluster_miu: ')
		print(cluster_miu)
		print('new_cluster_Sigma: ')
		print(cluster_Sigma)
		print('new_cluster_pi: ')
		print(cluster_pi)
		print('--------------------------')
		#print(member)
		#print(member_all)
		#print('UID: %d, cluster_pi: ' % _)
		#print(cluster_pi)

		# likelihood
		ll_new = np.sum(np.log(p_margin * 1e+1))
		'''
		ll_new = 0
		for i in range(n):
			s = 0
			for j in range(k):
				#print('cluster %d' % (j))
				s += cluster_pi[j] * gaussian(data[i], cluster_miu[j], cluster_Sigma[j])
				#print(cluster_pi[j])
				#print(gaussian(data[i], cluster_miu[j], cluster_Sigma[j]))
				#print('*********')
			#print('s: %d' % s)
			ll_new += np.log(s+0.01)
		ll = ll_new
		'''
		
		if(ll_new - ll < epsilon):
			flag = 0
			print('Converge at UID %d' % _)
			return (cluster_miu, cluster_Sigma, cluster_pi)
			exit(0)
		ll = ll_new
		
		

		# plot when debug
		if ( Debug and _ % debug == 0):
			if(d == 2):
				be = np.concatenate((data,r_),axis=1)
				plt.figure(figsize=(10,10)) 
				plt.scatter(be[:,0],be[:,1],marker = '+', c = be[:,2], cmap = cm, vmin = 0, vmax = k)
				plt.scatter(cluster_miu[:,0], cluster_miu[:,1],marker = 'o', color = 'r', s = 50)
				plt.show()
				#plt.savefig(str(_)+'.jpg')
			print('Uid: %d, ll: %f' % (_,ll))

	if flag:
		print('Reach the epoch number!')
		f.close()
		return (cluster_miu, cluster_Sigma, cluster_pi)

if __name__ == '__main__':
	x = np.array([5,5])
	miu = np.array([0,0])
	Sigma = np.eye(2) * 5
	print(5*Sigma)
	print(gaussian(x,miu,Sigma))
	a = np.array([[1,2],[3,4],[5,6]])
	print(np.sum(a, axis = 0))
	print(np.sum(a, axis = 1))
	print(np.log(x))
	print(np.exp(-845))
	data = np.array([[1,-2],[3,-4],[5,-6],[7,-8]])
	cluster_miu = np.array([5,-5])
	cluster_Sigma = np.zeros((2,2))
	r = np.array([0.2,0.2,0.2,0.4])
	for i in range(4):
		data_reg = np.add(data[i], -1*cluster_miu)[np.newaxis,]
		print(data_reg.T)
		S = np.dot(data_reg.T, data_reg)
		print(S)
		cluster_Sigma = np.add(cluster_Sigma, r[i] * S)
	#cluster_Sigma = cluster_Sigma[j] * (1.0 / member[j])
	print(cluster_Sigma)