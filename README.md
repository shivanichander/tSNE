# Visualisation of High Dimensional Data using tSNE – An Overview

We shall be looking at the Python implementation, and to an extent, the Math involved in the tSNE (t distributed Stochastic Neighbour Embedding) algorithm, developed by [Laurens van der Maaten](https://lvdmaaten.github.io/).

In machine learning problems, each feature of the elements in a dataset contributes a dimension. So, a dataset with many features is what we refer to as high dimensional data.

By ‘visualising data’, we mean plotting it on a graph, so that we can observe and make inferences or even judge the performance of our model. For visualising on a graph, we need to stick to two (planar graph) or three dimensions, because that is what the human eye processes, unlike computers which can process and work with many dimensions with ease. So, for us to visualise high dimensional data, we need dimensionality reduction techniques, to make the data presentable in a way that is well interpretable by us.

PCA is one such basic dimensionality reduction technique, where we do an orthogonal transformation, so that the resultant components are linearly uncorrelated, orthogonal and maximise the variance.

**tSNE is a more powerful technique that is capable of preserving the local structure as well as the global structure of the data.** That is, the aim of tSNE is to preserve as much of the significant structure in the high dimensional points as possible, in the low dimensional map.

Before looking at how tSNE achieves this, let’s understand SNE conceptually. For exact mathematical expressions and derivations, refer to the original paper [here](http://jmlr.csail.mit.edu/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).

## SNE (Stochastic Neighbour Embedding):
Hinton and Roweis came up with the SNE algorithm that converts high dimensional Euclidean distances between datapoints into conditional probabilities that represent similarities. The similarity between 2 datapoints is the conditional probability that one would pick the other as the neighbour, if neighbours were picked in proportion to their probability density under a Gaussian centred at one of them. So, for nearby points, this similarity is high, and vice versa, that is, similar points will be close by. The similarities are stored in a similarity matrix.

Likewise, the similarity can be modeled for the low dimensional counterparts too. If the projection from higher to lower dimension is correctly mapped, pairs of original points and their images will have same similarity.

This is the motivation behind the SNE algorithm. Lesser the mismatch between the two similarities, better is the model. 

The sum of Kullback Leibler divergences of the points is used as a measure of error. This is minimised using gradient descent.

## Physical Interpretation:
The paper gives a beautiful analogy of the gradient to be similar to the resultant force exerted on a set of springs between one point and all other points. The springs exert a force along the line joining the points and may be attractive or repulsive based on the distance between them. The force exerted is proportional to the length and stiffness (analogous to mismatch between the two sets of points).

So, SNE achieves equilibrium of the springs by attracting two map points if they are far apart while their data points are close and repelling if they are nearby while their corresponding data points are dissimilar.

## Why t distributed?
SNE faces the Crowding problem wherein there is lesser area in the two-dimensional map to accommodate the points from the higher dimensional space which results in a mismatch in the neighbourhoods between the two.

That is, even if a set of points are uniformly spaced in 10 dimensions, they tend to be more far away than actual when represented in 2 dimensions.

To overcome this, tSNE uses t-Student distribution for the map points, while a normal distribution is used for the data points.

Using this distribution leads to more effective data visualizations, where clusters of points are more distinctly separated.

## Notes on the Python Implementation:
The implementation of this algorithm can be easily done with one function, thanks to the [scikit-learn](http://scikit-learn.org) library!

It is the math behind it that requires to be appreciated and spent time with.

The datasest used in the code is the [UCI ML hand-written digits](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset which has approximately 180 images if each digit (a total of 1797 images) with 64 features (8x8 = 64 pixels).

The tsne function gives an effective projection into two-dimensional space that is visualised as shown in the result. We can see that the different digits have been classified well. Play around with the various parameters of the tsne function to get different outputs.

Running the same code again on the same inputs can give different outputs at different times. This is because t-SNE has a non-convex objective function which is minimized using a gradient descent that is initiated randomly. Based on the random state, the minima found differs.

To judge the performance of the model, we can simply look at the scatter plot generated to see if the clusters are well demarcated. However, for a more mathematical measure, we can compare the Kullback-Leibler divergences that t-SNE reports.

For larger datasets like [MNIST’s Handwritten digits](http://yann.lecun.com/exdb/mnist/) which has 70,000 images with 784 features [28x28 (784 pixels)], it is suggested to do a dimensionality reduction technique like PCA first, and then apply tSNE.

## References:
* http://jmlr.csail.mit.edu/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
* https://lvdmaaten.github.io/tsne/
* https://github.com/oreillymedia/t-SNE-tutorial

For **DataByte**
By **Shivani Chander**

Mentored by [Pragyaditya Das](https://github.com/Jeet1994)
