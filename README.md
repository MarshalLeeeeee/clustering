<h1>Clustering Algorithms In Python</h1>

<h3>test.py</h3>
<p>It is designed for testing the clustering algorithm.<br/>
Within this script, data is generated based on Gaussian distribution, where you can dicide the dimension, the number and the actual number of clusters of the dataset.<br/>
You can either give your mean and standard differance to the dataset, to make them randomly.<br/></p>

<h3>Kmeans.py</h3>
<p>
Use Euclidean distance as the measure of the distance, you can design other distance measurements as you wish.<br/>
You can use Para:slow to control the convergence speed of the algorithm.<br/>
You can use Para:Debug to dicide whether you wanna see the temporary result.<br/>
You can use Para:competitive to decide whether you want to use competitive algorithm to make adaption to the cluster_num automatically. The algorithm refers to RPCL, which gives the cluster a 'kick'. The algorithm uses the distance betwwen the cluster center and the point with high data density as the extent of 'kicking'.<br/> </p>

<h3>GMM.py</h3>
<p>Use GMM model to do clustering.<br/>
Add Para:independent to control the indeendency of the data in each dimension, default value is False.<br/>
You can use Para:independent = True to reach the variance between Kmeans and GMM.<br/></p>