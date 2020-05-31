# Gradient Boosting Alorithm Explained
---

## Overview:

Gradient boosting is used both for classification and regression(the approaches differ significantly). The Gradient boost creates a lot of weak lerners(small decision tress) in order to correctly classify the data in hand. It uses logit function in calculation for probability(please ensure you've read about logit function). Finlly, below are the steps Gradient boost follows to learn to predict a class

***Step1:*** Taking the complete data and calculating the log(odds) say there are 60 observations and out of which 40 are true/1 and 20 are false/0 so the log of odds will be log(40/20) = 0.693147180559945 or ~0.7

***Step2:*** After calculating the log odds, we put the value in a logit function and calculate the probability, say we get the probability value as 0.6. In all the ML algorithms, 0.5 is considered as threshold to classify records as 1 or 0. Here, since our probability is 0.6, we mark all records as 1 or true.

***Step3:*** Then we calculate the residuals i.e. observed value - predicted value. In our case, for all the records which were actually true, they will have a residual of 1-0.6 = 0.4 and for all the records which were false will have a residual of 0-0.6 = -0.6 

***Step4:*** Now, based on the residuals, we'll make a tree where instead of predicting 1 or 0 last leaves of tree will have residual values. Here, we should limit the number of leaves as there is always a chance to overfit, generally the values of # of leaves is restricted to 8 to 32.

***Step5:*** After calculating the probabilities and making a tree, we calculate output value of each leaf. This is done using the formula: 
					
          
             			sum(Residuals in the leaf)
		    ===========================================================
			Sum(Previous Probability * (1- Previous Probability)



Say in a leaf there was a 0 record and a 1 record, the residuals will be 0.4 and -0.6. Since, it is the 1st step, the probability was  0.6 after doing the math, we get (-0.2)/0.48 = -0.4167 . Similarly we calculate the output value for each leaf.

***Step6:*** Now we calculate log(odds) for each record with the above acquired values:

	New Log Odds = Previous Log Odds + Learning Rate * Output Value

Which equals to => 0.7 + 0.1 * -0.4167 = 0.6583. This becomes our new log odds and we again calculate the probability using the logit function. This becomes the new probability for that record.
Please note the learning rate is generally considered as 0.1.

***Step7:*** The same process continues till wee reach the max number of trees or the residuals become very small

***Step8:*** calculating the final probability for classification:
Say we had 2 trees as max limit, the final probability will be calculated as :

	Final Log Odds = initial log odds + LR * 1st tree's o/p value + LR * 2nd tree's o/p value

Feeding the log odds in logit function and getting the final probability for the record.
