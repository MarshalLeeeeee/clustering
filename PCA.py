import numpy as np

def norm(X):
	mean = np.mean(X, axis=1)[:,np.newaxis]
	X_norm = X - mean
	return (X_norm, mean)

def dist(w):
	return np.squeeze(np.dot(np.transpose(w),w))

def PCA(X):
	d = X.shape[0]
	n = X.shape[1]
	Sigma = np.dot(X, np.transpose(X)) / float(n)
	eigValue, eigMatrix = np.linalg.eig(Sigma)
	#print(eigValue,eigMatrix)
	X_ = np.transpose(np.dot(np.transpose(X),eigMatrix))
	print('initial data:')
	print(X)
	print('first principle component:')
	print(X_[0][:])


def SVD(X):
	d = X.shape[0]
	n = X.shape[1]
	U,S,V = np.linalg.svd(X)
	X_ = np.transpose(np.dot(np.transpose(X),U))
	print('initial data:')
	print(X)
	print('U,S,V:')
	print(U)
	print(S)
	print(V)
	print('first principle component:')
	print(X_[0][:])
	#return U[:][0]

def PCA_iter_major(X,lr=0.001,esp=1e-7):
	d = X.shape[0]
	n = X.shape[1]
	w = np.random.rand(d,1)
	Sigma = np.dot(X, np.transpose(X)) / float(n)
	alpha = 1e0
	loss = - np.dot(np.transpose(w), np.dot(Sigma,w)) / dist(w)
	#loss = - np.dot(np.transpose(w), np.dot(Sigma,w)) + alpha * np.power((np.dot(np.transpose(w),w) - 1),2)
	cnt = 0
	while(True):
		#print('cnt: ', cnt)
		#print('w: ', w)
		#print('loss: ', loss)
		deg =  (2 * np.dot(Sigma, w) * dist(w) - np.dot(np.transpose(w), np.dot(Sigma,w)) * 2 * w)/ np.power(dist(w),2)
		#deg = - (-2 * np.dot(Sigma, w) + 2 * alpha * (w - 1))
		w += lr * deg
		new_loss = -np.dot(np.transpose(w), np.dot(Sigma,w)) / dist(w)
		#print('new loss: ', new_loss)
		if(np.squeeze(loss) - np.squeeze(new_loss) < esp):
			break
		loss = new_loss
		cnt += 1
	w = w / np.power(dist(w),0.5)
	print('initial data:')
	print(X)
	print('direction:')
	print(w)
	print('first principle component:')
	print(np.dot(np.transpose(w),X))
	

if __name__ == '__main__':
	#X = np.transpose(np.array([[1.0,1.0],[1.0,-1.0],[-1.0,1.0],[-1.0,-1.0]]))
	X = np.transpose(np.array([[3.0,2.0],[1.1,0.8],[0.9,1.2],[-1.0,-0.0]]))
	X_normal, X_mean = norm(X)
	Sigma_norm = np.dot(X_normal, np.transpose(X_normal)) / float(X.shape[1])
	#PCA(X_normal)
	#print('---------------')
	#SVD(X_normal)
	#print('---------------')
	PCA_iter_major(X_normal)