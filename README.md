<h1>Clustering Algorithms In Python</h1>

<h3>test.py</h3>
<ul type = "circle">
<li><p>It is designed for testing the clustering algorithm.</p></li>
<li><p>Within this script, data is generated based on Gaussian distribution, where you can dicide the dimension, the number and the actual number of clusters of the dataset.</p></li>
<li><p>You can either give your mean and standard differance to the dataset, to make them randomly.</p></li>
</ul>

<h3>Kmeans.py</h3>
<ul type = "circle">
<li><p>Use Euclidean distance as the measure of the distance, you can design other distance measurements as you wish.</p></li>
<li><p>You can use Para:slow to control the convergence speed of the algorithm.</p></li>
<li><p>You can use Para:Debug to dicide whether you wanna see the temporary result.</p></li>
<li><p>You can use Para:competitive to decide whether you want to use competitive algorithm to make adaption to the cluster_num automatically. The algorithm refers to RPCL, which gives the cluster a 'kick'. The algorithm measures the distance betwwen the cluster center and the point with high data density as the extent of 'kicking'.</p></li>
<li><p>When you set Para:competitive to True, you can control the extent of adaption by setting Para:alpha and Para:beta.</p></li>
</ul>

<h3>GMM.py</h3>
<ul type = "circle">
<li><p>Use GMM model to do clustering.</p></li>
<li><p>You can use Para:Debug to dicide whether you wanna see the temporary result.</p></li>
<li><p>You can use Para:independent = True to reach the variance between Kmeans and GMM, where we modify the Sigma Matrix every iteration, but deem every dimension of data independent.</p></li>
</ul>