# Loss Functions in Deep Learning

## ***Information:***

The number of bits required to encode and transmit an event. Lower probability events have more information, higher probability events have less information.

Information = -log2(p)

## ***Entropy:*** 

Average # of bits required to transmit a randomly selected event from a probability distribution or 
It is the number of bits required to transmit a randomly selected event from a probability distribution.

 A skewed distribution has a low entropy, whereas a distribution where events have equal probability has a larger entropy.

H(X) = – sum (P(x) * log(P(x)))

## *Cross-entropy:*

It builds upon the idea of entropy from information theory and calculates the number of bits required to represent or transmit an average event from one distribution compared to another distribution.

P(x) is the target probability distribution and Q(x) is approximate probability distribution then cross entropy is calculated as:


H(P, Q) = – sum (P(x) * log(Q(x)))

Where P(x) is the probability of the event x in P, Q(x) is the probability of event x in Q and log is the base-2 logarithm, meaning that the results are in bits.

In classification, each example has a known class label with a probability of 1.0, and a probability of 0.0 for all other labels. A model can estimate the probability of an example belonging to each class label. Cross-entropy can then be used to calculate the difference between the two probability distributions.

The cross-entropy for a single example in a binary classification task can be stated by unrolling the sum operation as follows:

H(P, Q) = – (P(class0) * log(Q(class0)) + P(class1) * log(Q(class1)))

This looks like log loss

Suppose there are 100 training examples, there will be 100 predicted values, for each we'll calculate the cross entropy for every example and then will find the mean of that. We'll try to minimize the mean of cross entropy.

	• Cross-Entropy = 0.00: Perfect probabilities.
	• Cross-Entropy < 0.02: Great probabilities.
	• Cross-Entropy < 0.05: On the right track.
	• Cross-Entropy < 0.20: Fine.
	• Cross-Entropy > 0.30: Not great.
	• Cross-Entropy > 1.00: Terrible.
	• Cross-Entropy > 2.00 Something is broken

Cross-entropy will calculate a score that summarizes the average difference between the actual and predicted probability distributions for predicting class 1. The score is minimized and a perfect cross-entropy value is 0.

## *Hinge Loss:*

An alternative to cross-entropy for binary classification problems is the hinge loss function, primarily developed for use with Support Vector Machine (SVM) models.

It is intended for use with binary classification where the target values are in the set {-1, 1}.

The hinge loss function encourages examples to have the correct sign, assigning more error when there is a difference in the sign between the actual and predicted class values.

Reports of performance with the hinge loss are mixed, sometimes resulting in better performance than cross-entropy on binary classification problems.

Firstly, the target variable must be modified to have values in the set {-1, 1}.
Then the output layer activation should be updated as tanh

## *Squared Hinge Loss:*

The hinge loss function has many extensions, often the subject of investigation with SVM models.

A popular extension is called the squared hinge loss that simply calculates the square of the score hinge loss. It has the effect of smoothing the surface of the error function and making it numerically easier to work with.

If using a hinge loss does result in better performance on a given binary classification problem, is likely that a squared hinge loss may be appropriate.

Rest everything remains the same i.e. target variables should be -1, 1 and output activation should be tanh

## *Multi Class Entropy:*

Cross-entropy will calculate a score that summarizes the average difference between the actual and predicted probability distributions for all classes in the problem. The score is minimized and a perfect cross-entropy value is 0.

Cross-entropy can be specified as the loss function in Keras by specifying ‘categorical_crossentropy‘ when compiling the model.

## *Sparse Multiclass cross Entropy:*

A possible cause of frustration when using cross-entropy with classification problems with a large number of labels is the one hot encoding process.

For example, predicting words in a vocabulary may have tens or hundreds of thousands of categories, one for each label. This can mean that the target element of each training example may require a one hot encoded vector with tens or hundreds of thousands of zero values, requiring significant memory.

Sparse cross-entropy addresses this by performing the same cross-entropy calculation of error, without requiring that the target variable be one hot encoded prior to training.

Sparse cross-entropy can be used in keras for multi-class classification by using ‘sparse_categorical_crossentropy‘ when calling the compile() function.

The function requires that the output layer is configured with an n nodes (one for each class), with 3 different classes, we require three nodes, and a ‘softmax‘ activation in order to predict the probability for each class.


## *KL Divergence:*

KL Divergence for short, is a measure of how one probability distribution differs from a baseline distribution.

A KL divergence loss of 0 suggests the distributions are identical. In practice, the behavior of KL Divergence is very similar to cross-entropy. It calculates how much information is lost (in terms of bits) if the predicted probability distribution is used to approximate the desired target probability distribution.

As such, the KL divergence loss function is more commonly used when using models that learn to approximate a more complex function than simply multi-class classification, such as in the case of an autoencoder used for learning a dense feature representation under a model that must reconstruct the original input. In this case, KL divergence loss would be preferred. Nevertheless, it can be used for multi-class classification, in which case it is functionally equivalent to multi-class cross-entropy.

KL divergence loss can be used in Keras by specifying ‘kullback_leibler_divergence‘ in the compile() function.
We have to use one hot encoding for this.

---

[Click Here for more details on Loss Functions](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
