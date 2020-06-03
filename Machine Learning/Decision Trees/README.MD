# Decision Tree Explained

## Overview

A decision tree is a graphical representation of possible solutions to a decision based on certain conditions. It's called a decision tree because it starts with a single box, which then branches off into a number of solutions, just like a tree.

There are 3 main parts of the tree
- Root Node: First decision point or root of the tree
- Internal Nodes: Node is something which has splits, it can either split into nodes or leaves
- Leaf: Leaf is the last decision point i.e. we cannot break down the rules further

<p align="center">
<img src=https://github.com/amit-raj-repo/Data-Science-Python/blob/master/Machine%20Learning/Resources/DT-1.png width=400 height=300>
</p>

***Gini Impurity:***

Gini index or Gini impurity measures the degree or probability of a particular variable being wrongly classified when it is randomly chosen. But what is actually meant by ‘impurity’? If all the elements belong to a single class, then it can be called pure. The degree of Gini index varies between 0 and 1, where 0 denotes that all elements belong to a certain class or if there exists only one class, and 1 denotes that the elements are randomly distributed across various classes. A Gini Index of 0.5 denotes equally distributed elements into some classes.

Formula for Gini Index =
<p align="center">
  <img width="150" height="35" src="https://d1rwhvwstyk9gu.cloudfront.net/2019/04/Giniform-300x68.png">
</p>

## Steps followed by decision tree for classification

***Step1:*** Calculate gini index for all the variables present in the dataset

***Step2:*** The 1st split will be according to the variable with minimum gini index

***Step3:*** Now, for every node calculate the gini index again for all variables as well as independent leaf

***Step4:*** If the gini score for the node itself is the lowest score, we don't need to split it further and it can become a leaf node else pick the variable with minimum/lowest impurity value

***Step5:*** The process continues till we reach the ending criteria either min samples per leaf or max depth etc
