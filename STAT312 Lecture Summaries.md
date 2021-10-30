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
Uses a class dependant covariance matrix to create meaningful variables.
_f<sub>c</sub>(x) = P(x|Y=c)_

### Requirements:
- needs exclusively continuous predictors only
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
Uses a class dependant covariance matrix to create meaningful variables.
Instead of linear boundaries, QDA creates bounding curves

### Method:
Estimates _C<sub>p</sub> + (1/2) + C<sub>p</sub>(p+1) + C - 1_ parameters, which is more than LDA.

### Requirements:
- no.params < n
- _(1/2)*p(p + 1)_ < the number of instances in the clusters

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

### Features of Model Based Clustering:
As the clustering isn't based on proximity of clusters/points, or on density, the clusterings aren't necessary uniform or visually "logical"

# Networks and Connectivity