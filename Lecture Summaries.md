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
- Log transformations
- Dimension reduction

# Clustering 
### Similarity:
A measure of how different any two points are to each other, typically interpretted and determined through distance metrics

Typically requires the data to be scaled, such that all numeric values are on the same scale. 

### Feature creation:
Using the important features, we can create new variables that can be leveraged in clustering processes, or analysing the parameters of the clusters.

### Distance Metrics:
Requirement          | Equation
---------------------|------------------
Equality of distance | `Dist(x,y) == Dist(y,x)`
Difference in values means distance in space | `Dist(x,y) =/= 0`, if `x=/=y`
Triangular inequality | `Dist(x,y) + Dist(y,z) =/= Dist(z,x)`

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
### Classification Problems
Misclassification
- Sensitivity: what proportion of the true positives have we missed? `TP/(TP+FN)`
- Specificity: what proportion of the true negatives have we correctly identified? `TN/(TN+FP)`
- Receiver Operator Curves, ROC: TP relative to FP
    + ![Screenshot (1140)](https://user-images.githubusercontent.com/80669114/137035836-91e820d0-4a12-4ac3-87b7-61137f3a024a.png)
    + ![Screenshot (1141)](https://user-images.githubusercontent.com/80669114/137035926-98c481c7-5839-4265-982e-8735a0fd38c5.png)

Which of these matter more? Is the cost of one type of error higher than another?

### Cross Validation
### Boostrapping

# Decision Trees
### Governing principle:
Recursively splitting the dataset by the variable that increases the 'purity' of each node, relative to the parent node. 

Once a given purity or node size is reached, stop recursively splitting.

### Entropy:
`H(p) = -sum(P(s)-log(P(s)))`, where P(s) is the probability of being in state s.

Maximum entropy occcurs when, from all the information you have, the likelihood of a given instance being in any given state is equally likely.

### Information Gain:
We want to maximise the information gain that occurs at every itterative stage: the aim is to increase the reliability and accuracy of our predictions of where each instance should go. 

### Aggregate Models:
#### Bagging

#### Boosting

#### Random Forest


# Time Series
### Univariate data, what does this give us? 
- differences
- return periods
- seasonal trends
- anomaly detection
- summary statistics (mean, var, min, ...)

### What we don't have:
- Correlated factors and external event data
- No explanatories; we only have the response variable

### Exponential Smoothing:
Weights are given to all instances, and these weights decrease as the distance as the instance gets older (inversely proportional to difference between instance and most recent instance), at an exponential rate.

This has the effect of placing more emphasis on significant events and recent events.

### Equations
- Level: l<sub>t</sub> = alpha(Y<sub>t</sub> - S<sub>t</sub> - m) + (1- alpha) * (l<sub>t</sub> + b<sub>t-1</sub>)
- Trend: b<sub>t</sub> = B<sub>t</sub> (l<sub>t</sub> - l<sub>t-1</sub>) - (1 - B) * b<sub>t-1</sub>
- Seasonal: S<sub>t</sub> = Lambda (y<sub>t</sub> - (l<sub>t-1</sub>) - (b<sub>t-1</sub>)) + (1 - lambda) * S<sub>t-m</sub>, where `m` is the seasonal period

# Genetic Algorithms
Each individual is a fixed length reprentation of values that inform a strategy relating to a scoring heuristic. _"Most fit"_ individuals reproduce and push out _"less fit"_ individuals, promoting successful strategies, mimicing real world evolution.

### Fitness:
Fitness is defined as how well an agent's parameters allow it to solve the problem.

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
- Methods: proportional selection, tournament, ranked...

All selection methods are designed to create a biased selection criteria, biased towards selecting _"fitter"_ individuals over _"less fit"_ individuals.

### Crossover and mutation:
New instances are created by combining the values of multiple parents, and then subsequently applying mutation.
Crossover strategies | Specification
---------------------|-------------------
fixed point | [0,0,0,0] & [1,1,1,1] = [0,0,1,1]
multi point | [0,0,0,0] & [1,1,1,1] = [0,1,1,0]
uniform probability | fixed probability of a given value being crossed in from one parent, overriding another, applied to each gene

__Mutation:__ modifying the values of specific genes within an organism, based on probabilistic chance.
Can be changed to a random value, or to within a range of the original value 

### Issues with Genetic Algorithms
- This approach needs dynamic problems to be effective. It's very poor on simple regression and classification problems.
- Premature stabilisation/convergence: The selection processes will naturally push the population state to stability, thus requiring the mutation parameter to prevent convergence before a solution is found.
- Genetic Algorithms are not ensured to reach an optimal solution

### Assets of Genetic Algorithms
- Genetic Algorithms doesnt require a defined gradient/cost surface to be described before hand
- Genetic Algorithms are a highly flexible general purpose architecture
- It is highly applicable to complex multivariable environments

# Genetic Programming:

# Linear Regression
### MLR

### Splines and local models

# Multi Objective Optimisation

# Artifical Neural Networks

