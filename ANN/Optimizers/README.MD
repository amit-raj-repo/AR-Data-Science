# Optimizers in Deep Learning


We have a linear regression eq of:

## Gradient Decent
Y = mx + c

Let's try and optimize the c first, fixing m = 0.64, we initialize c as a random number or 0. We then calculate yhat for all the observations. Now, for a value of c we have y and yhat. In linear regression the loss function we have is the least squares one,

Loss = sum((yhat - y)^2)

For a c value say 0, we have the loss value 50. as we increase c, 0.25, 0.5, 1.0 et.c the loss value decreases. On further decrease, the loss value increases as the line moves away from the original data points.

To exactly know the c value with minimum loss, we need to calculate a lot of c values with decimals before reaching the minima. Gradient decent comes to rescue here, instead of having calculating 500 values of c between 0 and 1 or 2 or n it will move steeply for the points with high loss and take small steps when close to minima.

Here, if we see the loss function equation, it's a quadratic equation and graph of all quadratic equations is inverted glass. Or V/U shaped.

On, the U shaped curve of loss function, we start drawing tangents which is basically we start taking derivatives. In a U shaped curve if a tangent has slope = 0 we will get our minimum loss and optimum value for c.

GD, starts calculating the gradients or derivative, if the gradient value is far from 0, it will take large steps in c and if the value is close to 0 it'll take small steps to reach the minimum.
 
New intercept = old intercept - step size ( here step size is gradient or derivative * learning rate)

Or in Deep Learning terminology intercept/weights will be:w = w - a * dw

Now, there are high chances we do not get 0 gradient and can dangle between high decimal point c values i.e. 0.9999999 & 1. The GD here has a stopping criteria, when step size is less than 0.001 then the algo will stop and will return the corresponding c value.


The gradient decent work for a variety of loss functions and steps are common: 

***Step1 -*** Take the derivative of the loss function for each parameter, eg. Above we had beta values and intercept.

***Step2 -*** Pick random values of derivatives

***Step3 -*** plug random values in the derivative calculated in step 1

***Step4 -*** calculate step size as slope * learning rate

***Step5 -*** Calculate new parameters as:
		New Param = Old param - Step Size
		
***Step6 -*** Repeat steps 3 to 5 till the step size becomes very small or we reach the max number of iterations


## Variations of Gradient Decent (Stochastic and Mini Batch Gradient Decent):

In the above eg. We traversed the complete data to calculate the new step size, suppose we have 100 variables and 5 million rows, then the system will require a lot of calculations to calculate even one value of new param and will be time consuming and computationally intense. 

To tackle this we have 2 variations of gradient decent:

* Stochastic gradient decent:
		○ This takes 1 random training example to calculate New Param and process continues till we get the min loss value
* Minibatch Gradient Decent: 
		○ This divides data into batches i.e. say we have 1 million rows, we can divide this into minibatches of 1000 training examples and calculate New parameter using these 1000 randomly selected samples
		○ 1 mil /1000 = 1000 i.e. we'll divide data into 1000 small dataframes, calculate and optimize the loss

Both the SGD and Minibatch face a problem, they take some time to converge. Also, the normal gradient decent though slow will move steadily towards the minima while SGD and MBGD will have noise in it's path. Reason being the diversity of datapoints. In variations of GD, the gradient is calculated using single point or a minibatch of points, this can skew the calculations and instead of following a smooth downward trend, we'll move in a zig zag descending trend. 

To tackle this, we have SGD with Momentum. This focuses on horizontal movement instead of vertical movement.

We introduce a concept of momentum which is an exponential moving average of our previous weights. Earlier the new value of weights was calculated based on weights old - lr * dw now the lr * dw is replaced by a moving average of old dws as well this helps in smoothening the curve. 

New parameter = Old parameter - a * Vp (here p refers to the parameter)

Vp is the moving average of dws and is given by formula 

Vp = B * Vp + 1-B * dw

We just have to keep in mind that it takes and exponential moving average of all the previous dw to give the new dw. The B or the beta value is 0.9 mostly.

## AdaGrad Optimizer:

Adagrad is also called adaptive gradient decent. In the variations discussed above, we have seen that the learning rate remains constant. There have been studied which reveled if the the learning rate is set as constant the algorithm might never find the minima and will always have values closer to minima. 

If we change the learning rate such that with the # of epoch as the step size decreses, the learning rate also decreases. 

There are multiple ways of doing this, Adagrad uses the following formula to calculate the learning rate:

Learning rate after every pass = alpha / sqrt(( alphaT + epsilon)^2)

AlphaT = sum of squared values of previous dw

In the learning rate formula, we can clearly notice that denominator increases after every epoch and because of it the learning rate decreases.

The only disadvantage of this is sometimes the denominator value becomes vary high and make the learning rate close to zero which might result in high converging time.


## Adadelta & RMSprop Optimizers

In Adagrad, we saw the problem of AlphaT becoming pretty high, in these optimizers we take care of the problem by taking exponential weighted average.

AlphaT = beta_2 * weighted average of dw at t-1 + (1-beta_2) * (dw^2)

Learning rate after every pass = alpha / sqrt(AlphaT  + epsilon)

In the AlphaT equation we have a 1-beta_2 multiplicative factor which keeps the overshooting (dw^2) in control. Generally beta_2 value is around 0.95 or 0.99

## Adam's Algorithm:

Adam's algorithm is best of both worlds, here we use gradient decent with momentum and include weight decay as well
