# ADABoost Algorithm Explained
---

## Overview

AdaBoost is one of the first boosting algorithms to be adapted in solving practices. AdaBoost helps you combine multiple *Weak Classifiers* into a single *Strong Classifier*

## Comparison with Random Forest and Decision trees:

* Based on the variables we have, the algorithm creates various decision stumps i.e. 1 node and 2 leaves. This is basically a forest of stumps. As stumps can use only 1 variable to make decisions, they are technically weak learners.

* Unlike random forest where each tree has equal vote weight in final decision, some stumps have higher weight in making a decision in AdaBoost

* In RF each tree is independent of each other while in AdaBoost errors by 1st stump  influences by errors in 2nd stump and so on

## AdaBoost Steps for Classification: 

***Step1:*** Before the algorithm starts, we need to assign weights to the samples. If we have 10K records, the weight for the 1st pass will be same for each sample/record i.e. 1/10K. This means in the 1st pass all samples are equally important

***Step2:*** Now we travers through all variables and make decision stumps, 15 variables means 15 different stumps. If any column is numeric, we will search for the value with min error. Post this we calculate the Gini index for each stump. The stump with minimum Gini value will be the 1st stump in the forest.

***Step3:*** After deciding the 1st stump, we need to decide the say of this stump in final classification, remember some stumps have more say than others. The say for a stump is decided based on the formula below:Amount of say = 0.5 * log(1-Total Error / Total Error)

Here, Total Error is the sum of weights of the samples that the stump got wrong. E.g. in the above example there were 10K samples, each of the weight 1/10K and say the best stump made 2500 errors then the total error will be 2500/10000 = 0.25 and the amount of say will be ~0.23. The amount of say can take positive as well as negative values. -ve values say not of the event.

***Step4:*** Now we need to modify the sample weights so that next stump can take into account the learning from this stump. For the samples which are incorrectly identified, we'll increase the weight and for correctly identified sample we'll reduce the weight. The new weight for incorrectly identified samples would be:Weight for incorrectly identified = old sample weight * e^ amount of say
1/10K * e^0.23 = 0.0001 * 1.258

If the previous stump did a good job then amount of say will be higher which means the new sample weight will be much higher for incorrectly identified but if the stump did not do well, the new sample weight will have only marginal changes.

Weight for correctly identified = old sample weight * e^ - (amount of say)

***Step5:*** Now the sample weights are updated but if the sample weights should add to 1 so we normalize the new weights by dividing each value with the sum of weights.

***Step6:*** Now we randomly sample the records to create a new dataset, here any sample could be used twice. The data will be of the same size but records will be duplicated. This is done by randomly picking a number between 0 and 1, we have a column with cumulative sum of weights, we take the record/sample which is closest to the number. Since the weight of incorrectly identified samples is high, chances of them being selected is high

***Step7:*** Now, for the new data frame we provide equal weights and restart from step 1. Here we have multiple records of the incorrectly identified records now if the new stump incorrectly identifies the same record the penalty will be high and the new stump will be forced to perform poorly.

***Step7:*** Finally after making all the stumps, we sum the amount of say of each stump for that sample, if the sum of amount of say for 1 is higher than that of 0 then we classify it as 1.
 
