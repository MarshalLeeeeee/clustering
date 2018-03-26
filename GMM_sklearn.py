import numpy as np
import sklearn.mixture as mx
import matplotlib.pyplot as plt

def experimentAIC(data, save_dir, max_cluster_num = 7, save = True):

	if max_cluster_num > 10:
		max_cluster_num = 10
	if max_cluster_num <= 0:
		max_cluster_num = 7
	K = np.arange(max_cluster_num)
	types = ['spherical', 'tied', 'diag', 'full']
	colors = ['tomato','gold','limegreen','deepskyblue']
	cm = plt.cm.get_cmap('rainbow')
	aic = np.zeros((4,max_cluster_num))
	aic_min = float('inf')

	# find the overall best model
	for i in  range(4):
		for k in K:
			gmm = mx.GaussianMixture(n_components = k+1, max_iter = 100000, covariance_type = types[i])
			gmm.fit(data)
			aic[i][k] = gmm.aic(data)
			if aic[i][k] < aic_min:
				aic_min = aic[i][k]
				aic_best_gmm = gmm
				aic_best_index = [i,k]

	# find best full Gaussian model
	aic_min = float('inf')
	for k in K:
		if aic[3][k] < aic_min:
			aic_min = aic[3][k]
			aic_best_full_gmm = mx.GaussianMixture(n_components = k+1, max_iter = 100000, covariance_type = types[3])
			aic_best_full_index = [3,k]

	if(save):
		# aic graph
		plt.figure(figsize=(10,10))
		plt.title('aic_curve')
		plt.xlim((0, max_cluster_num+1))
		plt.ylim((np.min(aic) - 100, np.max(aic) + 100))
		rect = []
		for i in range(4):
			rect.append(plt.bar(K+(i*0.1 + 1)*np.ones(max_cluster_num), aic[i], width = 0.1, label = types[i], color = colors[i]))
		plt.legend(rect,types)
		plt.savefig(save_dir + 'aic_curve.png')

		d = data.shape[1]
		
		# plot the best model measured by aic
		aic_best_gmm.fit(data)	
		pred = np.array(aic_best_gmm.predict(data))[:,np.newaxis]
		if(d == 2):
			p = np.concatenate((data,pred),axis=1)
			plt.figure(figsize=(10,10))
			plt.title('aic_best_model')
			plt.title('best GMM measured by aic in model ' + types[aic_best_index[0]] + ' with ' + str(aic_best_index[1]+1) + ' clusters')
			plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = aic_best_index[1])
			plt.savefig(save_dir + 'aic_best_model.png')
		'''
		print('best GMM measured by aic in model ' + types[aic_best_index[0]] + ' with ' + str(aic_best_index[1]+1) + ' clusters')
		print(aic_best_gmm.means_)
		print(aic_best_gmm.covariances_)
		print('*******')
		'''

		# plot the best full model measured by aic
		aic_best_full_gmm.fit(data)	
		pred = np.array(aic_best_full_gmm.predict(data))[:,np.newaxis]
		if(d == 2):
			p = np.concatenate((data,pred),axis=1)
			plt.figure(figsize=(10,10))
			plt.title('aic_best_full_model')
			plt.title('GMM measured by aic in model ' + types[aic_best_full_index[0]] + ' with ' + str(aic_best_full_index[1]+1) + ' clusters')
			plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = aic_best_index[1])
			plt.savefig(save_dir + 'aic_best_full_model.png')
		'''
		print('best GMM measured by aic in model ' + types[aic_best_full_index[0]] + ' with ' + str(aic_best_full_index[1]+1) + ' clusters')
		print(aic_best_full_gmm.means_)
		print(aic_best_full_gmm.covariances_)
		print('*******')
		'''
	aic_best_full_gmm.fit(data)
	return (aic_best_full_gmm.means_, aic_best_full_gmm.covariances_)

def experimentBIC(data,save_dir,max_cluster_num = 7, save = True):
	if max_cluster_num > 10:
		max_cluster_num = 10
	if max_cluster_num <= 0:
		max_cluster_num = 7
	K = np.arange(max_cluster_num)
	types = ['spherical', 'tied', 'diag', 'full']
	colors = ['tomato','gold','limegreen','deepskyblue']
	cm = plt.cm.get_cmap('rainbow')
	bic = np.zeros((4,max_cluster_num))
	bic_min = float('inf')

	# find the overall best model
	for i in  range(4):
		for k in K:
			gmm = mx.GaussianMixture(n_components = k+1, max_iter = 100000, covariance_type = types[i])
			gmm.fit(data)
			bic[i][k] = gmm.bic(data)
			if bic[i][k] < bic_min:
				bic_min = bic[i][k]
				bic_best_gmm = gmm
				bic_best_index = [i,k]

	# find best full Gaussian model
	bic_min = float('inf')
	for k in K:
		if bic[3][k] < bic_min:
			bic_min = bic[3][k]
			bic_best_full_gmm = mx.GaussianMixture(n_components = k+1, max_iter = 100000, covariance_type = types[3])
			bic_best_full_index = [3,k]

	if (save):
		# bic graph
		plt.figure(figsize=(10,10)) 
		plt.title('bic_curve')
		plt.xlim((0, max_cluster_num+1))
		plt.ylim((np.min(bic) - 100, np.max(bic) + 100))
		rect = []
		for i in range(4):
			rect.append(plt.bar(K+(i*0.1 + 1)*np.ones(max_cluster_num), bic[i], width = 0.1, label = types[i], color = colors[i]))
		plt.legend(rect,types)
		plt.savefig(save_dir + 'bic_curve.png')

		d = data.shape[1]
		
		# plot the best model measured by bic
		bic_best_gmm.fit(data)	
		pred = np.array(bic_best_gmm.predict(data))[:,np.newaxis]
		if(d == 2):
			p = np.concatenate((data,pred),axis=1)
			plt.figure(figsize=(10,10))
			plt.title('bic_best_model')
			plt.title('best GMM measured by bic in model ' + types[bic_best_index[0]] + ' with ' + str(bic_best_index[1]+1) + ' clusters')
			plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = bic_best_index[1])
			#plt.show()
			plt.savefig(save_dir + 'bic_best_model.png')
		'''
		print('best GMM measured by bic in model ' + types[bic_best_index[0]] + ' with ' + str(bic_best_index[1]+1) + ' clusters')
		print(bic_best_gmm.means_)
		print(bic_best_gmm.covariances_)
		print('*******')
		'''

		# plot the best full model measured by bic
		bic_best_full_gmm.fit(data)	
		pred = np.array(bic_best_full_gmm.predict(data))[:,np.newaxis]
		if(d == 2):
			p = np.concatenate((data,pred),axis=1)
			plt.figure(figsize=(10,10))
			plt.title('bic_best_full_model')
			plt.title('GMM measured by bic in model ' + types[bic_best_full_index[0]] + ' with ' + str(bic_best_full_index[1]+1) + ' clusters')
			plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = bic_best_index[1])
			#plt.show()
			plt.savefig(save_dir + 'bic_best_full_model.png')
		'''
		print('best GMM measured by bic in model ' + types[bic_best_full_index[0]] + ' with ' + str(bic_best_full_index[1]+1) + ' clusters')
		print(bic_best_full_gmm.means_)
		print(bic_best_full_gmm.covariances_)
		print('*******')
		'''
	bic_best_full_gmm.fit(data)
	return (bic_best_full_gmm.means_, bic_best_full_gmm.covariances_)

def experimentVBEM(data, save_dir, max_cluster_num = 7):

	if max_cluster_num > 10:
		max_cluster_num = 10
	if max_cluster_num <= 0:
		max_cluster_num = 7

	concentration_types = ['dirichlet_process', 'dirichlet_distribution']
	weights = [0.001,0.01,0.1,1,10,100,1000]
	#weights = [1,10,100,1000]
	cm = plt.cm.get_cmap('rainbow')
	d = data.shape[1]

	for ctype in concentration_types:
		for w in weights:
			gmm = mx.BayesianGaussianMixture(max_cluster_num, max_iter = 100000, weight_concentration_prior_type = ctype, weight_concentration_prior = w)
			gmm.fit(data)
			pred = np.array(gmm.predict(data))[:,np.newaxis]
			if(d == 2):
				p = np.concatenate((data,pred),axis=1)
				plt.figure(figsize=(10,10))
				plt.title('VBEM_model with weight = ' + str(w) + ', in ' + ctype + ' mode with max cluster number' + str(max_cluster_num))
				plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2], cmap = cm, vmin = 0, vmax = max_cluster_num)
				#plt.show()
				plt.savefig(save_dir+'w-'+str(w)+'t-'+ctype+'.png')
	# plot the VBEM model

if __name__ == '__main__':
	print('a')