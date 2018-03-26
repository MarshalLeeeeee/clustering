import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def inputData():
	varList = raw_input('please input the target var: ')
	varList = varList.split(' ')
	X = []
	Y = []
	for i in varList:
		x = eval(i)
		max = 0
		dist = raw_input('please input the bound distance: ')
		dist = dist.split(' ')
		for y in dist:
			if (eval(y) > max):
				max = eval(y)
		X.append([x])
		Y.append([max])
	return (X,Y)

#X_train = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
#X_train = [[1],[3],[5],[7],[9]]
#X_train = [[1],[6],[11],[16],[21]]
X_train = [[0],[1],[2],[3],[4],[5],[8],[10]]

'''
# aic coef = 0
y_train = [[0.86156311111],
			[1.846378],
			[2.6896733333],
			[3.460204],
			[4.334179],
			[5.445660],
			[5.888299],
			[6.863427],
			[7.660400],
			[8.659627]
			]
			'''
'''
# bic coef = 0
y_train = [[1.134124],
			[2.256498],
			[3.457445],
			[4.610354],
			[5.626319],
			[6.715586],
			[7.867149],
			[9.286688],
			[10.439350],
			[11.302709]]
			'''
'''
# aic coef = 1
y_train = [ [1.029358],
			[2.7961331],
			[4.494385],
			[6.342523],
			[8.015663]]
			'''
			
'''
# bic coef = 1
y_train = [[1.112215],
			[2.930680],
			[5.426645],
			[7.875769],
			[10.066816]]
			'''
			

'''
# aic coef = 2
y_train = [ [1.870137],
			[2.807409],
			[4.591267],
			[6.522870],
			[8.219580]]
			'''
			
			
'''
# bic coef = 2
y_train = [[1.939106],
			[2.703692],
			[4.719621],
			[7.226597],
			[9.669441]]
			'''
			

'''
# aic coef = 3
y_train = [ [2.702176],
			[2.910139],
			[4.708362],
			[6.644575],
			[8.127383]]
			'''
			
'''
# bic coef = 3
y_train = [[2.863329],
			[3.142143],
			[4.620890],
			[6.514890],
			[8.821129]]
			'''

'''
# aic coef = 4
y_train = [ [3.550540],
			[3.904650],
			[4.438096],
			[6.461118],
			[8.302983]]
			'''
'''
# bic coef = 4
y_train = [[4.202384],
			[3.898001],
			[4.692106],
			[6.463342],
			[8.344564]]
			'''

'''
# aic coef = 5
y_train = [ [4.620140],
			[4.531622],
			[5.033166],
			[6.716657],
			[8.236996]]
			'''
'''
# bic coef = 5
y_train = [[5.299436],
			[4.469146],
			[5.035674],
			[6.442546],
			[8.421477]]
			'''

'''
# aic coef = 8
y_train = [ [7.431215],
			[8.339895],
			[10.209489],
			[14.929759],
			[20.238863]]
			'''
			
'''
# bic coef = 8
y_train = [[9.078177],
			[7.664369],
			[10.807439],
			[14.412600],
			[20.518492]]
			'''

'''
# aic coef = 10
y_train = [ [7.929200],
			[9.422547],
			[11.180476],
			[15.169809],
			[18.621284]]
			'''
			
'''
# bic coef = 10
y_train = [[11.322850],
			[9.113633],
			[10.749703],
			[15.062107],
			[19.836784]]
			'''

# aic_d1
y_train = [[0.84936523],
			[8.82625727e-01],
			[0.52310646],
			[0.24929964],
			[-0.19426909],
			[-0.3147164],
			[-0.08827831],
			[0.15620426]]
			

# aic_d2
y_train = [[0],
			[-6.67573214e-04],
			[0.02976109],
			[0.04799429],
			[0.07973368],
			[0.07856538],
			[0.03329007],
			[0.01756474]]
				

# bic_d1
y_train = [[1.14426119],
			[1.01801401],
			[0.31211393],
			[-0.03381533],
			[-0.41257584],
			[-0.74248054],
			[-0.38180221],
			[-0.58667511]]
			
			

# bic_d2
y_train = [[0],
			[0.01247005],
			[0.06870648],
			[0.07982327],
			[0.09550609],
			[0.11533546],
			[0.04428997],
			[0.04755463]]
			

#X_train, y_train = inputData()
X_test = [[2.25]]
y_test = [[3.0089176]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 22, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
print(regressor.coef_)
plt.figure(figsize=(10,10))
plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)

quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
print(regressor_quadratic.coef_)
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')
plt.show()
'''
print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)
print('1 r-squared', regressor.score(X_test, y_test))
print('2 r-squared', regressor_quadratic.score(X_test_quadratic, y_test))
'''