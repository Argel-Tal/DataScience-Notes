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

### Meta data:
Documents the purpose, limitations, units and sourcing of the dataset.

### Transformations:
- Scaling of the dataset (typically mean = 0 and standard deviation = 1)
- Time series: component vector transformations, through Fourier Transformations
- Log transformations: pushing non-linear variables onto linear scales
- Dimension reduction: has it's own selection

# Dimension Reduction


# Clustering 
### Similarity:
A measure of how different any two points are to each other, typically interpretted and determined through __distance metrics__

Typically requires the data to be scaled, such that all numeric values are on the same scale, normally to either `N(mean = 0, sd = 1)` or `[0:1]` for non-negative data. 

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
- As `kMeans works in all dimensions, and doesn't do dimension reduction, it can be prone to the issues of working in high dimensional space.

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
### Classification Problems: Misclassification - Is the cost of one type of error higher than another?
- Sensitivity: what proportion of the true positives have we missed? `TP/(TP+FN)`
- Specificity: what proportion of the true negatives have we correctly identified? `TN/(TN+FP)`
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
Growing many small trees that act in sequence, with each tree reducing the residual error passed to it from the previous tree. Each of these trees have a highly limited max depth.

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
### MLR

### Splines and local models

# Multi Objective Optimisation

# Artifical Neural Networks

