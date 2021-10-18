# Machine Learning processes:
### Traditional Statistical process:
        data problem ---> manually create model rules ---> evaluate model ---> communication
                               ↑                                 |
                               \---     analyse errors       <---/

### Machine Learning process:
        data problem ---> data collection ---> pre-processing and exploration ---> feature selection
                                    ↑                                                    |
                                     \                                                   |
                                      \                                                  ↓
                communication <--- Preformance analysis <--- Algorithm training <--- Model selection

### Type of Modelling
- Supervised Learning; known outputs
- Unsupervised Learning; clustering and grouping
- Semi-supervised
- Reinforcement
- Batch/Online
- Systems/Process design
- Instance Based
- Model Based

### Selection of Algorithms:
- What is the task/application?
- Do we need parameteric or non-parametric solutions?
- Do we need optimality?
- How long do we have?
- How much computational resources do we have?
- How big is the dataset?
- What is our preformance criteria, pure accuracy, false discovery, misclassification...?
- How important is explainability and simplicity?

# Data and it's properties
### Types of data:
- Simple
- Multidimensional
- Temporal
- Spatial
- Structured
- Unstructured
- Categorical
    + nominal
    + ordinal
    + interval
- Quantative
    + discrete
    + continuous

### Aspects of data quality
- Accuracy; does it match the "truth" of the real world environment
- Completeness; does it have missing values?
- Consistence; variance and outliers, does it make sense?
- Integrity; what is it's origin and fitness for the purpose?
- Timeliness; is it appropriate for when we're modelling this? Is there a lag, or has there been changes to the patterns between now and then?
- Accessibility; how easy is it to access and model with?

### Big Data - 4 V's
1. Volume: the amount of data
2. Variety: do we have data types, different values, different classes?
3. Velocity: how fast is the data being created?
4. Validity: do we trust the data? Does it fit the problem we're trying to solve?

### Resolving missing values:
- Imputation; replacing them with average values, or expected values given other variable states
    + Random imputation, random selection from other instances within the dataset
- Removal (column or row)

### Meta data:
Documents the purpose, limitations, units and sourcing of the dataset.

### Transformations:
- Scaling of the dataset (typically mean = 0 and standard deviation = 1)
- Time series: component vector transformations, through Fourier Transformations
- Log transformations: pushing non-linear variables onto linear scales
- Dimension reduction: reduce the complexity of the problem, and avoiding the boarding problem _"as dimensionality increases, the odds of an instance existing within at least one dimension's outlier range increases"_

# Dimension Reduction
### General Purpose:
- Simplification of modelling
- Reduces training time
- Simplifies model assessment
- Simplifies model communication

### Itterative Stepwise Selection (forward and backward)
A non-optimal way of itteratively removing/adding dimensions from the feature space until a specified complexity level is reached.
Ideally this ensures we only have the most impactful variables being retained into our modelling.

As these are stepwise processes, it only ensures we have a itteratively flattening gradient descent solution, not an overall minima/maxima. It can only ever find the best stepwise solution, it doesn't compute all possible combinations, only a variation of the previous version.

### PCA
__Motivations:__
- Reduce the number of dimensions
- Ensure the variables are independent of each other

__General Principal:__
The new axes, the principle components, are linear combinations of the original variables, such that the variance of the principal components decreases with each subsequent component and that all components are uncorrelated.

PCA does not utilize existing class labels (LDA does, and tries to ensure seperation between them)

Variables are standardised and centered to _N(mean = 0, sd = 1)_

From the `sum(var(principal components[Z]))`, we can see how much of the dataset's variance is explained, i.e. 
- "_the first two principal components account for 88% of the variability in the data; the first three principal components account for 97% of the variability_" 

Typically, the variance explained will decrease with each subsequent component, until it flatlines with no furter improvements to the amount of variation explained, producing an _"elbow plot"_:

![Screenshot (1151)](https://user-images.githubusercontent.com/80669114/137230407-e3ffdefc-24d8-43d7-9d6f-cff807cc21f2.png)

__Loadings:__
Z<sup>1</sup> = a<sub>1,1</sub>X<sub>1</sub> + a<sub>1,2</sub>X<sub>2</sub> + ... + a<sub>1,p</sub>X<sub>p</sub> where a values are the _loadings_ associated with each variable within that principal component. 

These _loadings_ are selected to minimise _var(Z<sub>1</sub>)_, and such that _a<sup>2</sup><sub>1,1</sub> + a<sup>2</sup><sub>1,2</sub> + ... + a<sup>2</sup><sub>1,p</sub> = 1_. The loading vectors of the principal components are thus the "direction" of the variation in the dataset.

__Sparse PCA__ is a version of PCA that penalises loading values, pushing them towards 0 as much as possible before preformance falls off.

### t-SNE: Stochastic Neighbour Embedding
Uses distance information and stochastic search to preserve neighbourhood relationships in high-dim space when shrunk onto a low-dim representation.

### Random Team selection:
Variables are binned into random teams/pools, which are then passed forward into models. The different teams are then assessed for their proportion of variation explained, or ability to inform a well generalising model.

It's an easy and quick way to do variable selection which provides good coverage but can be computationally expensive and it not guaranteed to be optimal.

# Optimisation & Scalability:
### Gridsearch:
Creates a "grid" across all variables, with an interval "step size" along the domain of that variable. We then step over that variable looking for model improvements.

As preformance improves, the step size decreases, allowing for more fine value finding. At too large of a step size, we may skip over optimal values. At too small of a step size, the algorithm will take too long.

### Loss Functions and Gradient Descent:
For problems without complex underlying functions we can find the the minimum value through derivative calculus. 

If the problem's function is unknown, we can evaluate the fitness value at various points, moving to decrease the cost/loss function until we reach a minima. By adding a momentum term, we can hopefully avoid local minima, and instead find the global minimum.

- __Gradient Descent optimasation:__ Δw<sub>i</sub> <- η∇w<sub>i</sub> E(w<sub>i</sub>), minimising yhat<sub>i</sub> - y<sub>i</sub>, where η is the _"learning rate"_.
- __Momentum enhanced Gradient Descent:__ Δw<sub>i</sub> <- γ*Δw<sub>i</sub> - η∇w<sub>i</sub> E(w<sub>i</sub>), minimising yhat<sub>i</sub> - y<sub>i</sub>, where η is the _"learning rate"_, and γ is the "momentum" coefficient.

![momentum1d](https://user-images.githubusercontent.com/80669114/137662969-72bb69de-8b4a-45b0-90ae-51aff1237ca4.gif)

_source: https://www.eleven-lab.co.jp/contents/wp-content/uploads/2019/08/momentum1d.gif_

# Clustering, Unsupervised Learning
### Similarity:
A measure of how different any two points are to each other, typically interpretted and determined through __distance metrics__

Typically requires the data to be scaled, such that all numeric values are on the same scale, normally to either `N(mean = 0, sd = 1)` or `[0:1]` for non-negative data. 

### Evaluation of Clustering: Silhuette Scores
Silhuette scores is the fit in a given cluster, compared to the fit of that instance into other clusters. 
Typically the sum/average of silhuette scores is evaluated across different clustering algorithms or across different numbers of clusters.

### Feature creation:
Using the important features, we can create new variables that can be leveraged in clustering processes, or analysing the parameters of the clusters.

### Distance Metrics:
Requirement          | Equation
---------------------|------------------
Equality of distance | `Dist(x,y) == Dist(y,x)`
Difference in values means distance in space | `Dist(x,y) =/= 0`, if `x=/=y`
Triangular inequality | `Dist(x,y) + Dist(y,z) =/= Dist(z,x)`

### kMeans
kMeans is a radial clustering algorithm centred around grouping points around an artifical cluster point (of which there are `k`). 

1. `k` Clusters are randomly intialised
2. Instances are allocated to one of the `k` clusters. 
3. Clustering points are placed at the mean (centre) of the instances allocated to them.
4. Points are reassigned to the cluster point they are closest to
5. This is then repeated till the algorithm stabilises.

__Drawbacks of kMeans__
- As this is a radially defined algorithm, it preforms poorly on shaped clusters
- `k` has be to specified apriori, meaning we need to run a number of trails at different levels of `k`
- As kMeans works in all dimensions, and doesn't do dimension reduction, it can be prone to the issues of working in high dimensional space.

As this is a deterministic algorithm, we can draw a classification boundary along the problem space, showing where the classification of a new instance would change. A larger `k` value is typically associated with better generation preformance, but if taken too high can comprimise preformance through underfitting.

![image](https://user-images.githubusercontent.com/80669114/137659841-d1634412-58bc-46d9-8ccd-00bfd9780c4b.png)

_source: https://brookewenig.com/img/KNN/DecisionBoundary.png_

### Dendrograms
Dendrograms are a clustering tree, where _like_ instances are grouped into _nodes_, which are then merge with other nodes.

The linkage metric can dramatically impact the structure of the resultant tree
Linkage type    | Description
----------------|-------------
Complete        | mimimises the furthest distance possible from the two nodes' values (min of A, max of B)
Average         | mimimises the difference in average (mean) value of each node
Minimum         | mimimises the difference in minimum value of each node
Maximum         | mimimises the difference in minimum value of each node
Ward             | minimises the sum of squares variance increase when two clusters are merged

### Kohonen Self Organising Maps (SOMs)
Visually storing similiar items together, designed to replicate neurological structuring patterns.

The SOM is a set of neurons, organised within a space (i.e. a position on a 2D plane), where there are is a neuron for each instance, comprised of a vector of weight values, corresponding to each variable. `shape(SOM) == shape(matrix[n,p])`
__Process__

1. Each instance is matched to a neuron
2. Update the neuron to be more like the matched instance, as defined by the learning rate
3. Update the surrounding neurons to be more like the matched instance of their shared neighbour, as defined by the neighbourhood size
4. reduce the learning rate: `lr`
5. reduce the neighbourhood size `influence with respect to distance betwen neurons: θ`

        [init] [init] [init]
        [init] [init] [init]
        [init] [init] [init]
             |
             ↓
        [init] [      init     ] [init]
        [init] [Δ0 d/d instance] [init]
        [init] [      init     ] [init]
             |
             ↓
        [Δ d/dΔ0] [Δ d/dΔ0] [Δ d/dΔ0]
        [Δ d/dΔ0] [   Δ0  ] [Δ d/dΔ0]
        [Δ d/dΔ0] [Δ d/dΔ0] [Δ d/dΔ0]

__neuron update rule:__ 
`w(t+1) = w(t) + θ(t) * lr(t) * (v(t) – w(t))`

At model start, the high `lr` and `θ` will mean neurons are changing highly to match instances, and also changing to respect far away points, which have a high influence on them.

In later epochs, neurons will be more resistant to change, with both instance values and neighbouring neurons exerting little influence on a neuron.

__Purpose of SOMs__
SOMs provide a way of visualising which instances tend to cluster together, suggesting a similarity of variable values.

# Measuring supervised model preformance
### Overfitting:
Machine learning models are essentially trying to memorise a dataset, subject to some constraints. In an ideal world, we would have all the data, and thus not need to create predictive/assumption based rules, we would just find a matching previously seen case. 

Our models will do this, if given enough freedom and flexibility. To prevent this memorisation and overfitting, we need to do early stop or penalisation. These constraining pressures force our models to develop useful predictive rules, which allow us to generalise onto previously unseen data.

We want to ensure that the error within our sample matches that of the wider population (out of sample error). Do do this, we reserve some data for training validation, called the _"testing"_ subset.

__"Bias:"__ model's reliance on existing samples, and the inaccuracy on previously unseen data: high bias is associated with poor preformance, __underfitting__.

__"Variance:"__ model's sensitivity to changes in the dataset -> how "jumpy"/noisey the model is

![image](https://user-images.githubusercontent.com/80669114/137659862-1d1e2ac1-f9fc-483b-84a1-bf6b3147d9ad.png)

_source: https://www.researchgate.net/publication/333827073/figure/fig1/AS:770834251644930@1560792615584/An-example-of-overfitting.png_

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
- Entropy: measurement of uncertainty

### Classification Problems: Misclassification - Is the cost of one type of error higher than another?

![image](https://user-images.githubusercontent.com/80669114/137656718-e41979e3-f6eb-4103-a61b-12de3e438a7b.png)

_source: https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg_

- Precision PPV: what proportion of the true positives have we missed? `TP/(TP+FP)`
- Sensitivity TPR: what proportion of the positive predictions were actaully positive `TP/(TP+FN)`
- Specificity TNR: what proportion of the true negatives have we correctly identified? `TN/(TN+FP)`
- False Discovery Rate FDR: What is the ratio of false positive to all positive predictions `FP/(FP+TP)`
- Receiver Operator Curves, ROC: TP relative to FP

No overlap, so no misclassification | Some overlap and misclassification
------------------------------------|--------------
![Screenshot (1140)](https://user-images.githubusercontent.com/80669114/137035836-91e820d0-4a12-4ac3-87b7-61137f3a024a.png) | ![Screenshot (1141)](https://user-images.githubusercontent.com/80669114/137035926-98c481c7-5839-4265-982e-8735a0fd38c5.png)

### Cross Validation
Cross validation is the process of reserving some of the data, keeping it out of the training subset, such that it can be used to validate the model's preformance, without it having influenced the model's values.

We do this, as with too much information models will eventually memorise the dataset, rather than developing useful predictive rules. Crossvalidation is done within the training-set, as it determines how the model develops, before the model is validated against the testing-set

# Decision Trees
### Governing principle:
Recursively splitting the dataset by the variable that increases the 'purity' of each node, relative to the parent node. 

Once a given purity or node size is reached, stop recursively splitting.

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

### Aggregate Models:
#### Bagging
Bootstrap aggregation, averaging a set of observations to reduce the variation in the model.
- regression: take the average
- Classification: apply the majority label (plurality)

We take B bootstrapped versions of the dataset from our training dataset, and build a decision tree on each. This produces B trees.
While this boosts preformance, we loose the advantage of trees being easy to interpret

#### Boosting
Growing many small trees that act in sequence, with each tree reducing the residual error passed to it from the previous tree. Each of these trees have a highly limited max depth. This is a form of __Stacking__, with the caveat that all stacked models are of the same type.

A learning rate `λ` is applied, enforcing how much of the previous tree is remembered when it's passed to the next tree (typically inversely proportional to the number of trees `B`)

Additionally, differnet trees can be given weights to different misclassification error types, such that different trees are penalised for failing to correctly classify different classes differently.

#### Random Forest
Bootstrapping both the instances __AND__ the variables available to each of the trees within the forest B, essentially an upgraded version of __bagging__.

These trees only consider a subset of the variables at each split, reducing the options they have available (available predictors labelled `m`, from all predictors `p`).

# Time Series
### Univariate data, "signal", what does this give us? 
- differences
- return periods: times between a repeated event
- General trend: long term direction of the variable's values
- seasonal patterns: a predictable length period where a regular pattern of change occurs (weekly, summer vs winter...)
- Cyclic patterns: similar changes to the variable's values, where those changes aren't over regular intervals/lengths (economic recessions)
- anomaly detection
- summary statistics (mean, var, min, ...)

### What we don't have:
- Correlated factors and external event data
- No explanatories; we only have the response variable

### Additive Model
Where the magnitude of the seasonal variation or variation around the trend __does not__ change with the raw value of the timeseries

__Level value:__ y<sub>t</sub> = S<sub>t</sub> + T<sub>t</sub> + E<sub>t</sub>

__Seasonally adjusted value:__ removing the influence of the season pattern, to see how the overall trend is changing, independant of those seasonal effects: y<sub>t</sub> - S<sub>t</sub>

variable      | description 
--------------|-----------------
y<sub>t</sub> | time series value
S<sub>t</sub> | seasonal component
T<sub>t</sub> | general trend
E<sub>t</sub> | error/noise

### Multiplicative Model
Where the magnitude of the seasonal variation or variation around the trend __changes__ proportion to the raw value of the timeseries.

__Level value:__ y<sub>t</sub> = S<sub>t</sub> * T<sub>t</sub> * E<sub>t</sub>

__Seasonally adjusted value:__ y<sub>t</sub>/S<sub>t</sub>

### Moving Average:
T<sub>t</sub> = 1/m * sum(y<sub>t+j</sub>), where `m = 2k + 1`

An estimate of the level at time `t`, is obtained by averaging values of the timeseries within `k` periods of `t`.

This somewhat smoothes out the random noise within the dataset.

### STL decomposition:
A version of additive model decomposition that allows for seasonal periods to change in size, meaning we're not limited to "real world periods" like months or days.

### Exponential Smoothing, kNN for Timeseries:
Weights are given to all instances, and these weights decrease as the distance as the instance gets older (inversely proportional to difference between instance and most recent instance), at an exponential rate.

This has the effect of placing more emphasis on significant events and recent events.

### Equations
__Holt-Winter's additive method:__
variable | equation | description
---------|----------|-----
estimated value | y<sub>t+h\|t</sub> = l<sub>t</sub> + h*b<sub>t</sub> + s<sub>t+h-m(k+1)</sub> |
Level | l<sub>t</sub> = α(Y<sub>t</sub>-S<sub>t</sub>-m) + (1-α) * (l<sub>t</sub>+b<sub>t-1</sub>) | weighted average between the seasonally adjusted observation, and the non-seasonal forcast
Trend | b<sub>t</sub> = B<sub>t</sub> * (l<sub>t</sub>-l<sub>t-1</sub>) - (1-B) * b<sub>t-1</sub> | weighted average of the estimate trend at `t`based on previous estimates of the trend
Seasonal | S<sub>t</sub> = `γ` (y<sub>t</sub> - (l<sub>t-1</sub>) - (b<sub>t-1</sub>)) + (1-`γ`) * S<sub>t-m</sub>, where `m` is the seasonal period and `γ` is the seasonal smoothing parameter | weighted average between the current seasonal index and the seasonal index of the same season last time 

__Autoregressive prediction:__ using the variable's past values to predict it's future values.

y<sub>t</sub> = c + Φ<sub>1</sub>y<sub>t-1</sub> + Φ<sub>2</sub>y<sub>t-2</sub> + ... + Φ<sub>p</sub>y<sub>t-p</sub> + E<sub>t</sub>, where p is the amount of previous instances we're using to predict future values.

__Autoregressive moving averages__ use forecast errors instead of levels:

y<sub>t</sub> = c + E<sub>t</sub> + θ<sub>1</sub>E<sub>t-1</sub> + θ<sub>2</sub>E<sub>t-2</sub> + ... + θ<sub>q</sub>E<sub>t-q</sub>

__Auto Regressive Intergrated Moving Average (ARIMA) models__:

Combining both the autoregressive model and the moving average created by the error terms:

y'<sub>t</sub> = c + Φ<sub>1</sub>y<sub>t-1</sub> + ... + Φ<sub>p</sub>y<sub>t-p</sub> + θ<sub>1</sub>E<sub>t-1</sub> + ... + θ<sub>q</sub>E<sub>t-q</sub> + E<sub>t</sub>

# Genetic Algorithms
### Evolutionary Strategies
Mutating computed parameters of a fitted function and selecting those subversions that worked better. Effectively a hill climbing algorithm, where it can't see the hill

Each individual is a fixed length reprentation of values that inform a strategy relating to a scoring heuristic. _"Most fit"_ individuals reproduce and push out _"less fit"_ individuals, promoting successful strategies, mimicing real world evolution.

### Fitness:
Fitness is defined as how well an agent's parameters allow it to solve the problem. 
A way of defining the fitness of a given individual is required to run and train a genetic algorithm. This is typically done as a "fitness function", evaluated at the end of a generation/epoch. 

    Initialisation 
    |   
    |       --- variation <--
    Evaluation              |
    |       --> selection ---
    |
    Termination

### Types of new instances:
- generational: whole population or a proportion of the population is replaced.
- steady state: newly created individuals replace single instance from within the population.

### Selection processes:
Methods: 
- __proportional selection:__ each individual is given a proportional chance of being selected, defined by it's fitness relative to the population's net fitness. `probSelection(i) = fitness(i)/sum(fitness(population))`
- __tournament:__ randomly select N individuals from the population and place them in a "pool". From that pool, we select the fittest individual and pass them forward into the next generation.
- __ranked:__ order the population by fitness. Parents are then selected from an upper proportion of this ranked list.

All selection methods are designed to create a biased selection criteria, biased towards selecting _"fitter"_ individuals over _"less fit"_ individuals.

### Crossover "recombination" and mutation:
New instances are created by combining the values of multiple parents, and then subsequently applying mutation.
Crossover strategies | Specification
---------------------|-------------------
fixed point | [0,0,0,0] & [1,1,1,1] = [0,0,1,1]
multi point | [0,0,0,0] & [1,1,1,1] = [0,1,1,0]
uniform probability | fixed probability of a given value being crossed in from one parent, overriding another, applied to each gene

__Mutation:__ modifying the values of specific genes within an organism, based on probabilistic chance.
Can be changed to a random value, or to within a range of the original value 

### Issues with Genetic Algorithms
- This approach needs dynamic problems to be effective. It's very poor on (logistic) regression and classification problems, as it needs to be able to assess the quality of the solution.
- __Premature stabilisation/convergence:__ The selection processes will naturally push the population state to stability, thus requiring the mutation parameter to prevent convergence before a solution is found.
- Genetic Algorithms are not ensured to reach an optimal solution
- __Drift:__ Given children are created from randomly selected parents, there is a chance for stochasticly induced drift away from optimal behaviour. Ideally, the fitness function should limit this, eliminating the _unfit_ individuals and rewarding a return to _fitter_ behaviour. 
- They're very slow, the random initialised behaviour, random selection, and lack of a strict gradient curve to move along means these algorithms can take a long time to convergence, if they do at all.
- As it's a random process, it's not directly reproducible, and small changes to random states and inital params can have a significant effect on the final outcome.

### Assets of Genetic Algorithms
- Genetic Algorithms doesnt require a defined gradient/cost surface to be described before hand
- Genetic Algorithms are a highly flexible general purpose architecture
- Genetic algorithm still function under unoptimised hyperparameters, it's fairly robust, it just won't be as efficent as it could be
- It is highly applicable to complex multivariable environments, where bruteforce search methods and are slow and expensive.

# Genetic Programming:
### Genetic Programming: Trees
Rather than having a fixed length reprentation (_chromosome_), tree-type genetic programming uses a variable length decision tree type structure to solve a problem. 

These have two fitness attributes:
- __Traditional fitness:__ accuracy of solution
- __Efficency:__ how long is the solution _"parismony"_

Evolutionary principles thus happen on the candidate programs themselves (the solution trees).
- Crossover is achieved by swapping branches of the various trees.
- Mutation is done by deleting and growing branches.

### Steps to Genetic Programming:
1. Define a set of terminals (explanatory variables and random constants)
2. Define a set of primative functions to be used within each branch 
3. Define a fitness function
4. Set hyperparameters
5. Set a termination criteria and a return structure, so you can get the solution(s) back out.

### Motivations of Genetic Programming:
- When explantory variables are interconnected in an unknown way
- When an approximate solution is sufficient _(always makes me uncomfortable, these tend to be the most permenant)_
- Areas where a solution architecture is unknown and undefined, but a fitness measure is defined.

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

# Linear Classification:
### General Principle:
Draw a dividing line that places a class on one side of the line, and another on the other side.

### Multiclass Classification:
This is done through a __one versus the rest__ approach, where each class has it's own binary model, where the output is `in this class || not in this class`.

Remainders which do not fall within the decision boundary of any class, in a shared zone between multiple classes, are allocated to the class who's decision boundary they are closest to.

![image](https://user-images.githubusercontent.com/80669114/137659886-70134a7b-7ce3-4282-ae37-e5951f8ffb21.png)

_source: https://raw.githubusercontent.com/satishgunjal/images/master/Binary_vs_Multiclass_Classification.png_

# Local Modelling
### General Practice:
- Use clustering to allocate local groupings
- Create a local model at each of a given set of local states
- Map new instances to one of the local groupings
- Apply respective local model to a new instance

### Error types:
- Error in the selection of which local model to use
    + This selection error `SE` is given by the difference between the error of the selected model `RE` and the error of the best model `BE`: SE<sub>i</sub> = RE<sub>i</sub> - BE<sub>i</sub>
    + Across the whole dataset, the selection error `SE` = `(1 / n) * sum (|(RE - BE) / RE|)`
- Error in the prediction of the local model - essentially normal regression/classification errors.

### Splines - Local Linear Models:
__Splines__ fit a linear model within a limited local domain, with the constraint that the change in gradient between domains == 0, and that the functions are continous (meaning they connect and don't have an aburpt change). This allows for higher flexibility over typical global-domain linear models, allowing us to better fit the dataset.

Splines can also be fitted with penalty terms to reduce the overfitting possible introduced through their increased flexibility (bias:variance trade off).

# Multi Objective Optimisation
### Objective Space:
Defined by the objective function, and the interactions of the cost criteria

This might be wanting to minimise and maximise a set of functions, subject so a set of constraints and cost functions (which serve as penalty terms for given solutions)

### Solutions:
__Dominance:__
A solution is said to dominate other solutions (`a>b`) if:
- For any one objective `a` is better than `b`
- For all remaining objectives `a` is no worse than `b`

Non-domainated solutions form the Pareto Front, a curve along which each solution is equally valid

![image](https://user-images.githubusercontent.com/80669114/137646377-f2f50ad4-a84d-4eef-872d-992fc20847a9.png)

### Evolutionary Multi-objective Optimisation (EMO)
__Concept:__ 
Searching for a covering along the Pareto-Front, not just a single solution/convergence.

Fitness is given by a the number of solutions that dominate another given solution (pair-wise comparisons), thus fitness can be expressed as ranked list of the population by dominance counts or by a frequency table.

__Convergence:__
To prevent standard Genetic Algorithmic convergence, we need to penalise proximity of solutions to each other (along inital/original dimensions). This helps to ensure we get a smooth covering along the entire Pareto-Front.

__Downsides of EMO:__ 
- Computational complexity: `O(N^2)`
    + Scalability ∝ number of objectives: `O(M*N^2)` where `M` is the number of objectives
- This still requires a post-hoc human lead selection of a final solution from the candidates along the Pareto-Front, and confidence that the algorithm has indeed found the furthest out front, not some local minimum.


# Artifical Neural Networks
### General Principles:
ANN work by trying to find the optimal weight values from a random seed value, through minimising error terms associated with the prediction `yhat`, using gradient descent:

        X0  ---> weight 0   --->  
        X1  ---> weight 1   --->  ┍------------------------┐
        X2  ---> weight 2   --->  | function ---> sigmoid  | ---> pred(Yhat)
        ...       ...             ┕------------------------┙
        Xn  ---> weight n   --->  

Δw<sub>i</sub> <- η∇w<sub>i</sub> E(w<sub>i</sub>), minimising yhat<sub>i</sub> - y<sub>i</sub>, where η is the _"learning rate"_.

__Backpropagation of residual errors:__ welcome to _the chain rule_

        Weights:      Δw = -η1 * 𝛿E/𝛿w
        Hidden Layer: Δv = -η2 * 𝛿E/𝛿v
                         = -η2 * 𝛿E/𝛿z * 𝛿z/𝛿v

### Activation functions and Hidden Layers
Adding these allows for increased complexity, extending the ANN beyond linear problems, allowing it to solve XOR type problems and fit polynomial functions curves

                Single Layer:
             X   X       O   X
             O   O       O   X

                Multi Layer:
        X   X       O   X       X   O
        O   O       O   X       O   X
        
![image](https://user-images.githubusercontent.com/80669114/137656751-71b85d23-e8bf-4e56-bccd-cf455e200a36.png)

_source: https://cdn-images-1.medium.com/max/1600/1*Gh5PS4R_A5drl5ebd_gNrg@2x.png_

__Types of Activation Functions:__
- Sigmoid (typical standardisation [0:1])
- RELU
- Softmax

# Network Modelling - Graph Theory
### Adjacency Matrix:
Adjacency Matricies are `N*N` matricies describing the number of vertices connected to each vertex/node.
Normally this is used to create the metric `p(k) ∝ k`

### Clustering Coefficient (C):
The clustering coefficient describes the density of connections around a given vertex/node.

The value is given by: `C = 2y/z(z-1)`

### Network Measures:
- Distance: The geodesic distance, the shortest path's length between two connected nodes
- Diameter: The maximum distance of all shortest paths between any two nodes 
    + _"the largest number of vertices which must be traversed in order to travel from one vertex to another when paths which backtrack, detour, or loop are excluded from consideration."_
    + source: _https://mathworld.wolfram.com/GraphDiameter.html_
- Closeness Centrality: inverted sum of the shortest distances between each node and every other node. 
    + This defines the "ease of access" across the network
- Between Centrality:   The ratio of all geodesic distances between pairs of nodes which run through other nodes.
    + This defines "how often does another node lie along the shortest path between other nodes on the network".


# Sensitivity Analysis: 
### Variation in the model's outputs ∝ variation in the model's inputs
- How does the model change under a variety of inputs?
- How does the value of a single variable influence the model? 
- How important is a specific input/instance to the model?

### Different Inputs:
Given a sample of `x` from a distribution, we might get a different distribution of `y`, based on the likelihood of the `x` values present in our sample being present within our dataset. _"Given that the model saw this data, which is has a `p(x)` chance of being present, how different is `y` from when the model saw this other distribution of data"_

### Noise:
If noise is introduced to a variable, how does that change the output of the model?

Doing this through the "hold all variables bar one fixed" may overlook correlated variables, which might introduce errors unrelated to noise itself. As such, we may want to use a more exhaustive searchtype.


# Ethical Concerns around Data Science:
### Delegation:
Where responsbility is delegated to algorithms, it's shortcomings and assumptions tend to be smoothed over in order to reduce workload, and get missed when handed to non-technical staff who haven't received sufficient training/information about the system. 

Not only does this mean the model may be applied where it isn't fit, or it's outcomes misinterpretted, it also becomeas a __blackbox__ risk: where the users of a ML system don't know how the algorithm came up with the solution it did, and have no way of breaking down the decision process.

### False Attribution:
The assumption that the system is infallible and unbiased allows for users to dismiss existing biases and to misrepresent the decisions of an algorithm. This capacity for deliberate or accidental misrepresentation is furthered when the system is not communicated to the public (i.e. propritory systems or national security)

### Do No Harm: 
Autonomous systems do have impacts on the lives of real people, their decisions shape the oppertunities and futures of those around them. Sometimes these algorithms may even shape the decision making processes of their users, shifting their world view and altering their decisions without explicitly making those decisions itself (i.e. social media "news" feeds).

### Proactive vs Reactive nature of data analysis:
Where Computer Science and Software has the explicit intent to create a solution before it is implemented, data science is often reactive, discovering patterns, trends or pathways in existing systems. It thus falls on the practitioners to selectively relay that information and apply it in an ethical way. 

Once discovered, the information and processes don't go away, and can often be extended beyond the initial scope. A facial recognition service for identifying your friends from a list in photos is just as much a criminal identification system, or a political dissident finder. 

### Data Collection:
- Is the data still being used for it's inital purpose? Is it being used in the way that participants consented to? 
- Was the data collected with informed consent at all?
- How long is the data being retained for? Did the provider expect you to have a record dating back that far? How valid is it to use data of a certain age, is it still appropriate (especially in a human context)?
- Is the data correctly anonymised? Is that even possible given the large array of explanatory variables?
- Is the data sufficiently protected, through security controls and encryption?

