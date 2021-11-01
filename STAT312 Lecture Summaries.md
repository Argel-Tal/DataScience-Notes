# Dimension Reduction
### Definitions:
- Original dimensionality: the original representation
- intrinsic dimensionality: actual dimensionality of the problem
- embedded dimensionality: the reduced space

### Multidimensional scaling:
__Definition:__
Maintaining the distance between observations without preserving the values of those observations.

1. Points close in original dim-space should remain close in the embedded dim-space
2. Points far from each other in the original dim-space should remain far away from each other in the embedded dim-space

This ensures we maintain the same clustering structures and patterns when moving into the embedded space.

### Closeness without Distance: 
- Metric MDS: preserving the original dimensionality as much as possible
- Non-metric MDS: preserving the ordering of distances; _3rd closest should still be the 3rd closest..._
- Model based: distances are essentially treated as compressable/expandable springs, exposed to _"stress forces"_

Examples:
1. Kruskal's non-metric NDS (isoMDS)
2. Sammon's non-linear MDS


# Principal Component Analysis:
### Base Principal Component Analysis:
__Purpose:__
Selecting a combination of variables and putting them into new variables, such that the combinations explain the most variation within the dataset.

__Properties of PCA:__
- They are linear combinations of _[X<sub>1</sub> : X<sub>p</sub>]_
- _variation(Z<sub>1</sub>) > var(Z<sub>2</sub>) > ...  > var(Z<sub>p</sub>)_, such that the cumlative variance sums to the amount of variance within the dataset (and thus gives us the proportion of the variance explained)
- All principal component are uncorrelated from each other

__Definition of a principal component:__
_Z<sub>1</sub> = a<sub>1,1</sub>X<sub>1</sub> + a<sub>1,2</sub>X<sub>2</sub> + ... + a<sub>1,p</sub>X<sub>p</sub>_, where the __loadings__ _[a<sub>1,1</sub>, a<sub>1,2</sub>, ..., a<sub>1,p</sub>]_ are chosen to maximise var(Z<sub>1</sub>), subject to the constraint that _a<sup>2</sup><sub>1,1</sub> + a<sup>2</sup><sub>1,2</sub> + ... + a<sup>2</sup><sub>1,p</sub> = 1_

To ensure that principal components are uncorrelated, a<sub>1</sub> and a<sub>2</sub> must be at right angles to each other. 

__Eigenvalues:__
Eigenvalues of PCA (λ) are the variances of a PC

__Limitations of PCA:__
- Can only find hyperplanes in the dataset
    + Thus, it cannot find non-linear structures
- Directions of variability =/= most interesting structural components

### Kernel PCA:
X = ZA, Z = XA<sup>T</sup>
Applies a _"kernel trick"_, such that we transform our X values using a kernel function, in order to provide PCA functionality on non-linear data
- typically a __gaussian radial basis function__, k(x<sub>i</sub>, x<sub>j</sub>) = exp(-σ|x<sub>i</sub> - x<sub>j</sub>|<sup>2</sup>), where if σ = 0 we're applying normal PCA. We want to optimise σ for the largest eigenvalues (λ), maximising the variance explained

__Limitations of Kernel PCA:__
- Can  be difficult to explain what the combinations in the PC's is, and thus what the relationships in the dataset are.
- Needs optimisation of σ, and thus multiple runs
- Gets ineffective w/ large n or p values, making it not useful for many instances, or highly dimensional data

### Sparse PCA
Assumes that most variables tend to be unimportant in explaining the dataset's trends, and that we should only be interest in a small subset of the variables. 

Thus, the PCA loadings should be == 0 where not necessary to explain the dataset. 

__Definitions:__
Sparse matrix _B; Z = XB<sup>T</sup>_
Default matrix _X = ZA_

We want to push down B, the number of non-zero loadings, and we do this by applying the combined penalty term: _α|B<sub>1</sub>| + β|B|<sub>2</sub><sup>2</sup>_

_min(A,B) |X - XB<sup>T</sup>A|<sup>2</sup> + α|B<sub>1</sub>| + β|B|<sub>2</sub><sup>2</sup>_. When α & β = 0 and B = A, it's normal PCA. The default values of α & β = 10<sup>-4</sup>

### PCA type overview

Feature |    PCA    | Kernel PCA | Sparse PCA
---|-----------|------------|-------------
__effective on low dimensional data__ | linear | non-linear | linear
__tunable params__ | no | yes (kernel function) | yes (tuning & penalty)
__interpretable PC's__ | maybe | no | maybe
__scalable to large _p_ or _n___ | somewhat | difficult | difficult

# Factor Analysis
Factor analysis is a model based approach to PCA, which assumes there is an underlying model/structure that explains the dataset's trends. As such, we can apply model evaluations to it's outputs.

__Purpose:__
1. Exploratory analysis: looking to see if there are latent factors in the dataset that aren't expressed within single variables
2. Confirmatory: testing whether there are relationships between variables and latent factors

### Method:
Variables must all be scaled to `μ = 0, σ = 1`

__With one factor (F):__
the underlying model is _X<sub>j</sub> = λ<sub>j</sub>F + ε<sub>j</sub>_, where  _μ(F) = 0 && σ(F) = 1_ and _μ(ε<sub>j</sub>) = 0_. _ε<sub>j</sub>_ captures the variance unexplained by our factor modelling.


The loadings of each factor are expressed as _λ<sub>j,1</sub>_, as they would be in PCA. 

With multiple factors, these factors are assumpted to be uncorrelated from each other.

__With 2+ latent factors:__
_λ<sub>j,1</sub><sup>2</sup> + λ<sub>j,2</sub><sup>2</sup> + ... + λ<sub>j,m</sub><sup>2</sup> + σ<sub>j</sub><sup>2</sup> = m_, where _m_ is the number of latent factors.

Variables _j & k_ are said to be highly correlated if they share high loadings on the same factors:
_corr(x<sub>j</sub>, x<sub>k</sub>) = (λ<sub>j,1</sub> - λ<sub>k,1</sub>) + (λ<sub>j,2</sub> - λ<sub>k,2</sub>) + ... + (λ<sub>j,m</sub> - λ<sub>k,m</sub>)_

__Selecting the number of latent factors (m):__
We keep adding new factors, until the additional σ contributed is greater than one (_σ > 1_), or until the cumlative variance explained reaches 80-90% (typically viewed on an elbow plot).

We can also rotate our factors through m-dimensions, till we find a solution where the maximum number of leadings are either `0|±1`, contributing a lot, or nothing.

### X<sub>j</sub> as a Random Variable:

F<sub>1</sub> is also a random variable, with a normal dist `μ = 0, σ = 1`, who's value is multiplied by the loading _λ<sub>j,1</sub>_

This allows each instance to be treated as an independent event/instance.

The loading is expected to fall in the domain `[-1,+1]`


# Linear Discriminant Analysis (LDA):
### Purpose: 
Can we know something about the values and features, given the allocation of instances into predetermined classes.
Uses a __class independant covariance matrix__ to create meaningful variables.
_f<sub>c</sub>(x) = P(x|Y=c)_

### Requirements:
- __needs exclusively continuous predictors only__
- class-independent covariance matrix Σ: an EEE model
- Posterior Probabilities: _p<sub>c</sub>(x) = π<sub>c</sub>f<sub>c</sub>(x) / Σ(π<sub>c</sub>f<sub>c</sub>(x))_
- Prior Probabilities: _π<sub>c</sub> = P(Y=c)_

### Estimated Parameters:
Parameter | Calculation | Definition
----------|-------------|-----------
π<sub>c</sub> | n<sub>c</sub>/n | the proportion of each class within the larger dataset
μ<sub>c</sub> || sample mean of each class
Σ<sub>c</sub> || the weighted average of sample covariances of each class
n estimated params  | c<sub>p</sub> + 1/2 * p(p+1) + C -1 | needs to be significantly below n

### Maximising discriminant functions:
- _δ<sub>c</sub>(x) = (X<sup>T</sup>Σ<sup>-1</sup>μ<sub>c</sub>) - (1/2) * (μ<sub>c</sub><sup>T</sup>Σ<sup>-1</sup>μ<sub>c</sub>) + log(π<sub>c</sub>)_
- _δ<sub>c</sub>_ divides the space into sections by class, creating planes/boundaries
- ![Axes-1-and-2-from-linear-discriminant-analysis-LDA-based-on-17-brittle-stars](https://user-images.githubusercontent.com/80669114/139508260-067db2f0-ff54-4215-8ff1-c760f66eb8c3.jpg)
    + source: https://www.researchgate.net/figure/Axes-1-and-2-from-linear-discriminant-analysis-LDA-based-on-17-brittle-stars_fig3_330571683

# Quadratic Discriminant Analysis:
Uses a __class dependant covariance matrix__ to create meaningful variables.
Instead of linear boundaries, QDA creates bounding curves

### Method:
Estimates _C<sub>p</sub> + (1/2) + C<sub>p</sub>(p+1) + C - 1_ parameters, which is more than LDA.

### Requirements:
- no.params < n
- _(1/2)*p(p + 1)_ < the number of instances in the clusters
- class-dependent covariance matrix Σc: a VVV model

# Independant Component Analysis:
Assuming that X = n.p is a matrix of indpendant signals; _S<sub>1</sub>, S<sub>2</sub>, ... , S<sub>p</sub>_, we can define the matrix as:

X<sub>1</sub> |  a<sub>1,1</sub>S<sub>1</sub> + | a<sub>1,2</sub>S<sub>2</sub> + | ... | a<sub>1,p</sub>S<sub>p</sub>
-------|----------|--------|--------|--------
X<sub>2</sub> |  a<sub>2,1</sub>S<sub>1</sub> + | a<sub>2,2</sub>S<sub>2</sub> + | ... | a<sub>2,p</sub>S<sub>p</sub>
...|...|...|...|...
X<sub>n</sub> |  a<sub>n,1</sub>S<sub>1</sub> + | a<sub>n,2</sub>S<sub>2</sub> + | ... | a<sub>n,p</sub>S<sub>p</sub>

_X = SA<sup>T</sup>, S = XA_, where A is an "unmixing" matrix

__Motivations:__
ICA does not assume normality, and instead uses an approximation to the true entropy formula, approximating the true distribution using the observed distribution.

ICA is used in scenarios with multiple input streams, such as audio and medical imaging.

### Whitening:
Using singular value decomposition, such that we can assume _var(x<sub>j</sub>) = 1_ and _corr(x<sub>j</sub>, x<sub>k</sub>) = 0, if j =/= k_:

- _X = UDV<sup>T</sup>_
    + _K = sqrt(n) * V * D<sup>-1</sup>_
    + _est(X) = X * K_
    + _cov(est(X)) = (1/n) * est(X)<sup>T</sup> * est(X) = 1_


# Model Selection and Bootstraps
Using previous data to evaluate the likelihood of outputs being the values they are, given previously seen input-output values. 

Likelihood Ratio Test Statistic (LRTS) = 2(l(G<sub>1</sub>)-l(G<sub>0</sub>)), where l is the log-likelihood evaluated at the estimated GMM parameters.
LRTS typically has a χ<sub>2</sub> distribution. This can be tested for with a bootstrapped distribution, evaluating the number of time the LRTS<sub>b</sub> exceeded the expected value LRTS<sub>0</sub>. This allows us to test if we need to reject the current number of clusters in our model.

1. _H<sub>0</sub>: G = G<sub>0_
2. _H<sub>1</sub>: G = G<sub>1</sub> > G<sub>0_

### Quantifying Uncertainty: 
Bootstraping to compute the std error and CI's
- Let θ denote the parameter of estimates/model
- Using a boostrap we estimate θ<sub>b</sub> for the dataset X<sub>b</sub>
- Compute the covarience matrix of θ<sub>b</sub>, who's diagonal elements are the std error of the parameters θ<sub>b</sub>
    + These should be normally distributed.
- Can also plot the distribution of sample means

### Covariance matricies:
- i,j<sup>th</sup> component sum _= cov(i,j) = mean(x<sub>i</sub>, x<sub>j</sub>) - mean(x<sub>i</sub>) * mean(x<sub>j</sub>)_, a measure of how much i and j co-vary.
    + σ<sup>2</sup><sub>i</sub> = var(x) = cov(i,j)
    + corr(i,j) = cov(i,j) / (σ<sub>i</sub> * σ<sub>j</sub>) 

### Geometry of covarience matrices:
- Volume: how widely spread the contours are
- Shape: how circular/spherical the contours are
- Orientation: the direction of the contours

![E1XbL](https://user-images.githubusercontent.com/80669114/139516478-1656fe22-56e7-4da5-b4fa-533a629153d0.png)

image source: https://i.stack.imgur.com/E1XbL.png

### Features of Model Based Clustering:
As the clustering isn't based on proximity of clusters/points, or on density, the clusterings aren't necessary uniform or visually "logical"


# Networks and Connectivity
### Measures of Connectivity:
- Spectral Clustering: opperates on the principle of points being "connected", rather than them being "close" to one another.
- Connectedness: the principle of being able to traverse along a network's edges, to get between two nodes/verticies.
- Adjacency Matrix (A): encodes the information about which nodes are connected to which other nodes; _A<sub>i,j</sub> == 1_ if the verticies _i & j_ are connected, and _== 0 otherwise_.

                (1)         
               /   \            [0, 1, 1]
              /     \       A = [1, 0, 1]
            (3) ---- (2)        [1, 1, 0]

- The graph Laplachin (L) connects along the diagnonal; L<sub>ii</sub> = -(no. of edges coming from node i). Using this, we can find connected clusters: if the vertex is in in a column, then _C<sup>i</sub>==1_, otherwise _C<sup>i</sub>==0_

                (1)         
                   \            [-1, 1,  0]
                    \       L = [1, -1,  0]
            (3)     (2)         [0,  0,  0]

C<sub>1</sub> = [1,1,0]<sup>T</sup>, C<sub>2</sub> = [0,0,1]<sup>T</sup>. In cluster one, vertices 1 & 2 are connected, but vertex 3 is not. The opposite is true for cluster two.

### Generalising A and L:
__Affinity Matricies:__
_A<sub>i,j</sub> = k(x<sub>i</sub>, x<sub>j</sub>)_, where _k_ is a kernel function, represents a modified adjacency matrix, referred to as an affinity matrix.

_L<sub>i,i</sub> = A<sub>i,i</sub> - sum<sub>j</sub>(k(x<sub>i</sub>, x<sub>j</sub>))_

### Spectral Clustering Process:
- Choose a kernel function
- General graph Laplachin
- Find the columnwise vectors _C<sub>1</sub>, C<sub>2</sub>, ..., C<sub>m</sub>_, such that intracluster Laplachins are 0, and there are less clusters than vertices _(m < n)_
- Preform kMeans clustering on the binary columnwise vectors _C_

### Neighbourhoods Ν(i)
Things near a point (i), can be defined either as the k-closest points to i, or those points within a distance from i.

__Isometric feature mapping: isomap__
These measure dissimilarity _d<sub>i,j</sub>_ between the points _i_ and _j_. 

__Process:__
1. define a neighbourhood metric
2. create a network by connecting edges between points and those within their neighbourhood
3. compute the dissimilarity _d<sub>i,j</sub>_, by _sum(dissimilarity)_ along the shortest path between points _i_ and _j_
4. Apply Multi-dimensional-scaling maintaining _d<sub>i,j</sub>_, with a specified number of new embedded dimensions.

__Local Linear Embedding:__
1. define a neighbourhood metric
2. find an approximation of each point using the weighted sum of their neighbours
3. using these weights, find the coordinates Z<sub>i</sub> which satisfy approximations in a pre-determined lower dimension space.

# Selecting Cluster Counts
Plotting intracluster variation, we should see a rapid decrease in variance as clusters increase, however the gradient of this decrease should rapidly fall off to near horizontal. This will have an "elbow" point, where our optimal number of clusters is achieved, before we risk overfitting.
### Gap Statistic
The Gap Statistic is an attempt at automatically locating the optimal number of clusters (_no.clusters = k_), by measuring gradient change

Statistic| Calculation | Definition
---------|-------------|--------
_gap(k)_ | _= (1/B) * sum(log(w<sup>b</sup><sub>k</sub> - log(w<sub>k</sub>)))_ | The difference in the (log) total within cluster variation from the actual data vs the expected (log) within cluster variation from uniform/reference data.

- with _current no.clusters (k) < ideal no.clusters (k*)_, the current clusters are likely to not be uniform in internal variance, so the gap statistic will grow rapidly

- with _k* < k_ the clusters are becoming more uniform, and thus the gap statistic will fall off.

![image](https://user-images.githubusercontent.com/80669114/139561539-59411f68-d0bd-4297-8b4d-02be40dc23e6.png)

Image source: _http://www.sthda.com/sthda/RDoc/figure/clustering/determining-the-number-of-clusters-gap-statistic-k-means-1.png_

### Silhuette Score; Clustering evaluation
metric                                 | definition
---------------------------------------|-----------
`a(i)`                                 | the average dissimilarity of an instance (i) to all observations in a given cluster
`b(i)`                                 | the average dissimilarity of an instance (i) to all observations in all other clusters
`s(i) = (b(i)-a(i)) / max{a(i), b(i)}` | bounded `[-1,+1]`, where `+1` is ideal
`S = (1/n) * sum(s(i))`                | Silhuette score

# Hyperplane clustering and Support Vectors:
### Maximal Margin Classifiers:
The maximal margin classifier is the hyperplane which provides optimal class seperation. The _"margin"_ is defined as the minimum distance of an observation to the dividing hyperplane.

Ideally, we want to maximise the "no-man's land", created by the margin, between classes that are split by the hyperplane.

This requires scaling, such that all dimensions are on the same scale, and so we can use distance metrics. 

The instances which inform the margin are refered to as _"support vectors"_, as they are vectors of length p, and which support the creation of the margin:

### Support Vectors:
Support vector based classifiers allow for a soft-margin, where a small number of instances are allowed to fall on the wrong side of the margin/hyperplane. 

The degree to which instances are allowed to cross the margin/hyperplane is determined by the tuning parameter _C_, which is the cost of incorrect classifiations/margin-violations. 
A higher C will produce a model which is more tolerant of misclassification (_1/C_).

The evaluation of a given plane is done by summing the classification error ε<sub>i</sub>, and testing that against C; _sum(ε<sub>i</sub>) ≤ C_ , where ε<sub>i</sub> is defined as follows:

value                   | condition
------------------------|-----------
ε<sub>i</sub> = 0       | instance is on the correct side of BOTH the classifier plane AND the margin
0 < ε<sub>i</sub> < 1   | instance is on the correct side of the hyperplane, but within the margin
1 < ε<sub>i</sub>       | instance is on the WRONG side of the hyperplane, ε<sub>i</sub> increases as it moves AWAY from the hyperplane

If the sum comparison against C holds, then the hyperplane is valid.

### Defining the hyperplane:
The hyperplane is defined as _f(x) = B<sub>0</sub> + sum(α * k(x<sub>i</sub>, x<sub>j</sub>))_, where _k(x<sub>i</sub>, x<sub>j</sub>)_ is a kernel function, allowing us to create non-linear hyperplanes.

__Example Kernel functions - _k(x<sub>i</sub>, x<sub>j</sub>)_:__
1. Linear: inner product _(x<sub>i</sub> ⋅ x<sub>j</sub>)_
2. Polynomial: _(C = γ * x<sub>i</sub> ⋅ x<sub>j</sub>) <sup>d</sup>_, where _d>1_ and _C_ is a tuning parameter
3. Gaussian radial: _exp(-γ * | x<sub>i</sub> - x<sub>j</sub> | <sup>2</sup>)_
4. Laplachin: _exp(-γ * | x<sub>i</sub> - x<sub>j</sub> |)_

### Multiclassification vs Binary Classification:
In __[1 : 1]__ classification a classifier is created for each class, so for _k_ classes, each class is compared against _k-1_ classifiers, creating _(1/2) * k(k-1)_ classifier comparisions overall.

For __[1 : all]__ classification we fit k SVM's and compare one class (+1) against the rest, which are all pooled together (-1). Instances are then classified as belonging to the class who's classifier they scored highest on.

# Logistic Regression:
Logisitic Regression is binary classification where _p<sub>1</sub> = p_ is the probability of an instance belonging to class 1, and _p<sub>0</sub> = 1 - p_ is the probability of the instance belonging to the opposite class.

As this is deterministic, we can draw a classifier boundary through several dimensions: _η(x) = 0_

### Multiclass Logistic Regression: 
With _k_ classes, for _k = 1, 2, k-1_ we use 
- _p<sub>k</sub> = p<sub>0</sub> * exp(η<sub>k</sub>(x))_
- p<sub>0</sub> = 1 / (exp(η<sub>1</sub>(x) + exp(η<sub>2</sub>(x) + ... + exp(η<sub>k-1</sub>(x)))

Class 0 is the baseline class, used as a reference point. From there we allocate the instances to the classes who they have the highest probability of belonging to (_"softmax"_). For _k_ classes, there will be _k_ values, each corresponding to a class, where the sum of those values = 1:
- σ(z)<sub>k</sub> = 
- p<sub>k</sub> = σ(1, η<sub>1</sub>, η<sub>2</sub>, ..., η<sub>k-1</sub>)<sub>k</sub>

# Linear Regression
### Multiple Linear Regression:
- single predictor:     `yhat = w[0]*x[0] + b`
- multiple predictors:  `yhat = w[0]*x[0] + w[1]*x[1] + ... + w[p]*x[p] + b`
- polynomial predictors: `yhat` = Θ<sub>0</sub> `+ sum`(Θ<sub>i</sub> * z<sub>i</sub>), where z = [x<sub>1</sub>, x<sub>2</sub>, x<sub>1</sub>x<sub>2</sub>, x<sub>1</sub><sup>2</sup>...]

This essentially gives the output as a weighted sum function, where weights are applied to each of the explanatory variables.

The weight and offset (`b`) values are chosen such that they reduce the mean-square-error (MSE).
These can be set at different levels for each instance; 
- Random slope: different weight levels for each instance, will produce a different curve
- Random offset: different offset/error term for each instance, will set the origin at a different point

### Penalised Linear Regression:
__Lasso Regression - L1:__
Weights are penalised such that some weights are force to be exactly 0, ensuring we only have the predictors that are asboutely necessary. This provides a form of implicit variable selection. As such, lasos is preferable over ridge regression for datasets that are highly dimensional

The degree of penalisation is controlled by the alpha term: `α`, where higher `α` pushes a greater amount of weights to be == 0

Lasso regression function: `yhat = α[0]*w[0]*x[0] + α[1]*w[1]*x[1] + ... + α[p]*w[p]*x[p] + b`

__Ridge Regression - L2:__
Weights are penalised such that higher values are penalised more, promoting weights to have a low value, close to 0, thus influencing the slope as little as possible. This helps to prevent terms from having unnecessarily outsized values, where other predictor's have weights to cancel out the affects of the outsized ones.

The degree of penalisation is controlled by the alpha term: `α`, where higher `α` gives greater penalisation

Ridge regression function: `yhat = α[0]*w[0]*x[0] + α[1]*w[1]*x[1] + ... + α[p]*w[p]*x[p] + b`

__Elastic Net:__
An elastic net essentially merges the penalisation of both Lasso and Ridge regression into one equation. The proportion of which penalisation term is applied is controlled through a hyper parameter, which splits the penalisation between [0:1] such that `prop(L1) + prop(L2) == 1`

# Evaluating model preformance: 
### Overfitting:
Machine learning models are essentially trying to memorise a dataset, subject to some constraints. In an ideal world, we would have all the data, and thus not need to create predictive/assumption based rules, we would just find a matching previously seen case. 

Our models will do this, if given enough freedom and flexibility. To prevent this memorisation and overfitting, we need to do early stop or penalisation. These constraining pressures force our models to develop useful predictive rules, which allow us to generalise onto previously unseen data.

We want to ensure that the error within our sample matches that of the wider population (out of sample error). Do do this, we reserve some data for training validation, called the _"testing"_ subset.

__"Bias:"__ model's reliance on existing samples, and the inaccuracy on previously unseen data: high bias is associated with poor preformance, __underfitting__.

__"Variance:"__ model's sensitivity to changes in the dataset -> how "jumpy"/noisey the model is

### Components of model assessments:
- cross model comparisons
- cross dataset comparisons (testing and training data splits)
- sensitivity analysis; how the model changes when it's inputs change __covered further in lecture 25__

### Regression Metrics:
- Mean Squared Error
- Root Mean Squared Error
- Mean Absolute Error
- Root Relative Squared Error
- Coefficient of determination (R<sup>2</sup>)

### Classification Metrics:
- Confusion matrix [on target vs off target classifications]
- ROC
- AUC
- Entropy/Deviance: measurement of uncertainty

### Classification Problems: 
__Misclassification - Is the cost of one type of error higher than another?__

![image](https://user-images.githubusercontent.com/80669114/137656718-e41979e3-f6eb-4103-a61b-12de3e438a7b.png)

_source: https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg_

- __Precision__ PPV: Of the posititve predictions, how many are true? `TP/(TP+FP)`
- __Sensitivity/Recall__ TPR: what proportion of the positive predictions were actaully positive `TP/(TP+FN)`
- __Specificity__ TNR: what proportion of the true negatives have we correctly identified? `TN/(TN+FP)`
- __False Discovery Rate__ FDR: What is the ratio of false positive to all positive predictions `FP/(FP+TP)`
- __F score__: `2 / ((1/precision) + (1/recall))`
- __Receiver Operator Curves__, ROC: Binary classification - True Positive Rate relative to False Positive Rate, and where cut points will cause FNs and FPs
- __AUC curves__: shows the area underneath the ROC curves - ideally will be a horizontal asymptote, not a positive diagonal (given by random allocation)

No overlap, so no misclassification | Some overlap and misclassification
------------------------------------|--------------
![Screenshot (1140)](https://user-images.githubusercontent.com/80669114/137035836-91e820d0-4a12-4ac3-87b7-61137f3a024a.png) | ![Screenshot (1141)](https://user-images.githubusercontent.com/80669114/137035926-98c481c7-5839-4265-982e-8735a0fd38c5.png)

# Classical Statistics vs High Dimensional Data
Methods like Least Squares preform poorly in high dimensional spaces, given the over abundance of parameters to fit on, and the increasing likelihood of a point being an outlier in at least one dimension. Thus, the models produced using these methods will be prone to overfitting.
The core issue of this is _"noise"_, it gets harder to isolate meaningful information from random noise as dimensions are added. 

Further, it becomes increasingly likely that we could overlook important variables, interpretting them as combaintions of other variables; detecting spirulus correlations.

### Binary classiciation problems:
- Deviance = _-sum(Y<sub>i</sub> * log (π<sub>i</sub>) + (1 - Y<sub>i</sub>) * log(1 - π<sub>i</sub>))_, where Y<sub>i</sub> == 1 if π<sub>i</sub> >= t, and Y<sub>i</sub> == 0 if π<sub>i</sub> < t
    + typically t==0.5, but if the cost of False Positives =/= the cost of False Negatives, this can be modified to respect the cost imbalance.
- Misclassification = sum(Y<sub>i</sub> =/= est(Y<sub>i</sub>))


# Generalised Additive Models (GAMs):
Y<sub>i</sub> = β<sub>0</sub> + sum(f<sub>j</sub>(X<sub>i,j</sub>)) + ε<sub>i</sub>, where f<sub>j</sub> are non-linear functions of a single variable.

This allows us to fit non-linear functions to each feature, without needing to manually apply transformations.
The model being "additive" also allows us to preform "change in x[,j], holding x[,-j] constant" type assessments.

GAMs are the generalised format, which contains Splines.

### Splines - Local Linear Models:
__Splines__ fit a linear model within a limited local domain, with the constraint that the change in gradient between domains == 0, and that the functions are continous (meaning they connect and don't have an aburpt change). This allows for higher flexibility over typical global-domain linear models, allowing us to better fit the dataset.

Splines can also be fitted with penalty terms to reduce the overfitting possible introduced through their increased flexibility (bias:variance trade off).

- Natural Splines are defined by the restriction that their 2nd derivative == 0
    + _d<sup>2</sup>/dx<sup>2</sup> = 0_
- Smoothing Splines apply a penalty term to gradient change (Δm): 
    + _penalty = λ*∫(g'' (t)<sup>2</sup> * dt)_

# Decision Trees:
### Governing principle:
Recursively preforming a binary splitting the dataset by the variable that increases the 'purity' of each node, relative to the parent node. 
Once a given purity or node size is reached, stop recursively splitting.

### Preventing overfitting:
Tree penalties are applied as _Score = RSS + penalty term_:
_Score = sum(sum((Y<sub>i</sub> - est(Y<sub>i</sub>) <sup>2</sup>) + α |T|))_, where _T_ is the no. nodes ("_leaves_") on the tree, and _α_ is the penalty term (_α = 0_, no penalty on size)

### Entropy:
`H(p) = -sum(P(s)-log(P(s)))`, where `P(s)` is the probability of our instance being in state s.

__Entropy of the system of subsets S<sub>i</sub>:__ 
H(S, at) = sum(<sup>k</sup><sub>i=1</sub>P(S<sub>i</sub>) * H(S<sub>i</sub>)), where the subsets S<sub>i</sub> are `i = 1,...,K`

Maximum entropy occcurs when, from all the information you have, the likelihood of a given instance being in any given state is equally likely.

__Entropy for positive and negative cases:__
H(S<sub>i</sub>) = -p<sub>i</sub><sup>+</sup> * log(p<sub>i</sub><sup>+</sup>) - p<sub>i</sub><sup>-</sup> * log(p<sub>i</sub><sup>-</sup>)

### Information Gain:
We want to maximise the information gain that occurs at every itterative stage: the aim is to increase the reliability and accuracy of our predictions of where each instance should go. 

This is defined as the decrease in entropy generated by that split. `I(S, at) = H(S) - H(S, at)`

To find the best split, we find the split that provides the maximum information gain value. 

### Advantages and Disadvantages of Tree type models
Advantages       | Disadvantages
-----------------|---------------
easy to communicate | low preformance on their own
graphically simple  | small changes to the data can dramatically alter the tree structure
mirrors human decision making processes | prediction surfaces aren't smooth functions, but blocks
can handle both regression and classification |



### Aggregate Models:
#### Bagging
Bootstrap aggregation, averaging a set of observations to reduce the variation in the model.
- regression: take the average
- Classification: apply the majority label (plurality)

We take B bootstrapped versions of the dataset from our training dataset, and build a decision tree on each. This produces B trees.
While this boosts preformance, we loose the advantage of trees being easy to interpret

The out-of-bag error or OOB error is the error on those instances not present within each of the booststrapped dists. A typical bootstrapped sample will involve 2/3 of the instances, and the remaining 1/3 are _"out-of-bag observations"_

#### Boosting
Growing many small trees that act in sequence, with each tree reducing the residual error passed to it from the previous tree. Each of these trees have a highly limited max depth. This is a form of __Stacking__, with the caveat that all stacked models are of the same type.

A learning rate `λ` is applied, enforcing how much of the previous tree is remembered when it's passed to the next tree (typically inversely proportional to the number of trees `B`)

Additionally, differnet trees can be given weights to different misclassification error types, such that different trees are penalised for failing to correctly classify different classes differently.

#### Random Forest
Bootstrapping both the instances __AND__ the variables available to each of the trees within the forest B, essentially an upgraded version of __bagging__.

These trees only consider a subset of the variables at each split, reducing the options they have available (available predictors labelled `m`, from all predictors `p`).
