import numpy as np

def input():
	x = raw_input('input the data')
	x = x.split(' ')
	print(x)
	y = []
	for i in range(len(x)):
		y.append(eval(x[i]))
	return y

def gauss(arr):
	data = np.array(arr)
	mean = np.mean(data)
	data_norm = data - mean
	var = np.sum(np.power(data_norm, 2))
	return (mean, var)

m, v = gauss(input())
print(m)
print(v)