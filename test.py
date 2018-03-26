import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture as mx
import os
import json
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

def center(num, distance):
	angle = 2.0 * np.pi / num
	mean = []
	for i in range(num):
		theta = angle * i
		mean.append([np.cos(theta) * distance, np.sin(theta) * distance])
	return mean

def sigma(num, s, var = 0):
	sig = []
	for i in range(num):
		sig.append([[s,var],[var,s]])
	return sig

def toTuple(arr):
	ans = []
	num = len(arr)
	for i in range(num):
		ans.append((arr[i][0],arr[i][1]))
	return ans

def toList(arr):
	ans = []
	num = len(arr)
	for i in range(num):
		ans.append([arr[i][0],arr[i][1]])
	return ans

def toMeanList(arr):
	ans = []
	for i in range(len(arr)):
		ans.append([arr[i][0], arr[i][1]])
	return ans

def toSigmaList(arr):
	ans = []
	for i in range(len(arr)):
		s = []
		s.append([arr[i][0][0],arr[i][0][1]])
		s.append([arr[i][1][0],arr[i][1][1]])
		ans.append(s)
	return ans

def testBound(kp, number, k_num, distance, save_dir):

	for _ in range(5):
		var = 5 * _ + 1
		print('var: %d' % var)
		adistance = 0
		bdistance = 0
		for i in range(10):
			distancel = 0.7
			distancer = 50.0
			epsilon = 1e-5
			while(distancer - distancel > epsilon):
				distance = (distancel + distancer) / 2.0
				sig = sigma(k_num, var)
				mean = center(k_num, distance)
				data, miu, Sigma, partial = init(2,number,k_num,s = sig, m = mean, kp = kp1000)
				d = data.shape[1]-1
				data = data[:,:d]
				am, ac = GMM_sklearn.experimentAIC(data,save_dir,10,save = False)
				oms = np.array(toTuple(mean), dtype = [('x', float),('y',float)])
				ams = np.array(toTuple(am), dtype = [('x', float),('y',float)])
				oms = np.sort(oms,order=['x','y'])
				ams = np.sort(ams,order=['x','y'])
				oms = np.array(toList(oms))
				ams = np.array(toList(ams))
				aerror = np.sum(np.power((ams - oms) / distance, 2)) if len(ams) == len(oms) else float('inf')
				if(aerror < float('inf')):
					distancer = distance
				else:
					distancel = distance
			print('aic bound distance: %f' % distancer)
			adistance += distancer

			distancel = 1.0
			distancer = 50.0
			epsilon = 1e-5
			while(distancer - distancel > epsilon):
				distance = (distancel + distancer) / 2.0
				sig = sigma(k_num, var)
				mean = center(k_num, distance)
				data, miu, Sigma, partial = init(2,number,k_num,s = sig, m = mean, kp = kp1000)
				d = data.shape[1]-1
				data = data[:,:d]
				bm, bc = GMM_sklearn.experimentBIC(data,save_dir,10,save = False)
				oms = np.array(toTuple(mean), dtype = [('x', float),('y',float)])
				bms = np.array(toTuple(bm), dtype = [('x', float),('y',float)])
				oms = np.sort(oms,order=['x','y'])
				bms = np.sort(bms,order=['x','y'])
				oms = np.array(toList(oms))
				bms = np.array(toList(bms))
				berror = np.sum(np.power((bms - oms) / distance, 2)) if len(bms) == len(oms) else float('inf')
				if(berror < float('inf')):
					distancer = distance
				else:
					distancel = distance
			print('bic bound distance: %f' % distancer)
			bdistance += distancer
		print('aic average bound distance: %f' % (adistance / 10.0))
		print('bic average bound distance: %f' % (bdistance / 10.0))

def measureError(kp, number, k_num, distance, var):

	sig = sigma(k_num, var)
	mean = center(k_num, distance)
	data, miu, Sigma, partial = init(2,number,k_num,s = sig, m = mean, kp = kp1000)
	save_dir = './demo/compare/n' + str(number) + '-k' + str(k_num) + '-d' + str(distance) + '-s' + str(var) + '/'
	if (not os.path.exists(save_dir)):
		os.mkdir(save_dir)

	n = data.shape[0]
	d = data.shape[1]-1
	target = data[:,d,np.newaxis]
	data = data[:,:d]
	cm = plt.cm.get_cmap('rainbow')
	cluster = np.array(mean)
	if(d == 2):
		p = np.concatenate((data,target),axis=1)
		plt.figure(figsize=(10,10))
		plt.title('origin')
		plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = 5)
		plt.scatter(cluster[:,0], cluster[:,1],marker = 'o', color = 'r', s = 50)
		plt.savefig(save_dir + 'origin.png')

	am, ac = GMM_sklearn.experimentAIC(data,save_dir,10)
	bm, bc = GMM_sklearn.experimentBIC(data,save_dir,10)
	oms = np.array(toTuple(mean), dtype = [('x', float),('y',float)])
	ams = np.array(toTuple(am), dtype = [('x', float),('y',float)])
	bms = np.array(toTuple(bm), dtype = [('x', float),('y',float)])
	oms = np.sort(oms,order=['x','y'])
	ams = np.sort(ams,order=['x','y'])
	bms = np.sort(bms,order=['x','y'])
	oms = np.array(toList(oms))
	ams = np.array(toList(ams))
	bms = np.array(toList(bms))
	aerror = np.sum(np.power((ams - oms) / distance, 2)) if len(ams) == len(oms) else float('inf')
	berror = np.sum(np.power((bms - oms) / distance, 2)) if len(bms) == len(oms) else float('inf')

	with open(save_dir + "result", 'w') as f:
		f.write('am: ')
		f.write(json.dumps(toMeanList(am)))
		f.write('ac: ')
		f.write(json.dumps(toSigmaList(ac)))
		f.write('bm: ')
		f.write(json.dumps(toMeanList(bm)))
		f.write('bc: ')
		f.write(json.dumps(toSigmaList(bc)))
		f.write('oms: ')
		f.write(json.dumps(toMeanList(oms)))
		f.write('ams: ')
		f.write(json.dumps(toMeanList(ams)))
		f.write('bms: ')
		f.write(json.dumps(toMeanList(bms)))
		f.write('aerror: %f' % aerror)
		f.write('berror: %f' % berror)

def testVBEM(kp, k_num, number, distance, var):
	sig = sigma(k_num, var)
	mean = center(k_num, distance)
	data, miu, Sigma, partial = init(2,number,k_num,s = sig, m = mean, kp = kp1000)

	n = data.shape[0]
	d = data.shape[1]-1
	target = data[:,d,np.newaxis]
	data = data[:,:d]
	cm = plt.cm.get_cmap('rainbow')
	cluster = np.array(mean)	

	for k in range(10):
		save_dir = './demo/compare/VBEM-n' + str(number) + '-k' + str(k) + '-d' + str(distance) + '-s' + str(var) + '/'
		if (not os.path.exists(save_dir)):
			os.mkdir(save_dir)
		if(d == 2):
			p = np.concatenate((data,target),axis=1)
			plt.figure(figsize=(10,10))
			plt.title('origin')
			plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = 5)
			plt.scatter(cluster[:,0], cluster[:,1],marker = 'o', color = 'r', s = 50)
			plt.savefig(save_dir + 'origin.png')
		GMM_sklearn.experimentVBEM(data, save_dir, k+1)



if __name__ == '__main__':
	
	kp1000 = [333,333,334]
	number = 1000
	k_num = 3
	distance = 5
	var = 1
	save_dir = ''

	#testBound(kp1000, number, k_num, distance, '')
	testVBEM(kp1000,k_num,number,distance,var)
	
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
	
	

