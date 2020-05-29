#Principal Component Analysis (PCA) Explained:
---

***Step 1:*** Say we have 2 columns and 50 observations. We will plot these 2 columns and measure the average for variable 1 and same for variable 2 (Just assume variable 1 is at x axis and variable 2 on y axis). 

***Step 2:*** From this average value we can measure the center of the data i.e. we draw a line from average of var1 and var2, the intersection of the 2 line is our center of data.

***Step 3:*** Now we consider the center of data as origin and shift the data. Note: we are only shifting the origin all the points remain at the same point/distance from each other

***Step 4:*** Now we take a random line to start with, Note: the line will always pass through origin. 

***Step 5:*** For the random line we project out points to the random line, similar to how we do in linear regression. After projecting the point on the line, we measure the distance between the projected point and the origin. This is done for each projected point. 

***Step 6:*** After getting all the distance between projected points and origin, we square them and add them up. We then continue till we get the line with maximum/largest squared distance between projected points and origin.

Here maximum/largest distance implies the best fit in linear regression, in regression we minimize the squared distance between points and line, here we maximize the distance between projected points and origin. These 2 however, are the same things.

The line with maximum/largest squared distance between projected points and origin is called PC1 or Principal component 1

***Step 7:*** Now we measure the slope of the best fit line. Say we got a slope of 0.25

y= mx as c =0 from here we come to know that if we move 4 units in x direction, we'll move 1 unit in y direction. This means the variation in x is much higher than in y.

***Step 8:*** Now, we calculate the singular/unit/eigen vector. We make a triangle with x axis value as 4, y axis value as 1 and the hypotenuse value will be calculated using Pythagoras theorem. We get the value as 4.12. This is the distance that the best fit line covers from the origin, if we move 4 units in x direction and 1 unit in y direction. Now to make it a unit vector, we need to divide everything by 4.12. After doing that we get x value as 0.97 and y value as 0.242. This means if we move a unit distance from the origin, we move 0.97 units in x and 0.242 units in y. The unit vector we move on the best fit line is called Eigen/Singular vector for PC1. The proportions here i.e. 0.97 and 0.242 are called loading scores.

The maximum value of squared distance between projected points and origin is called the Eigen value of PC1 and if we take a square root of that then its called the singular value for PC1.

***Step 9:*** Now we find PC2, which is nothing but a perpendicular line on PC1 passing through origin. As it is the perpendicular line, we get the slope as 4 hence, for singular vector for PC2, we move 0.97 units in y direction and -0.242 in x direction.

***Step 10:*** Now we turn the PC graph such that PC1 is horizontal. For calculating total variation in the data, we sum Eigen value 1 + Eigen value 2 and divide by n-1. Here lets assume the 

Variation in PC1 = EV/n-1 = 15 and 
Variation in PC2 = EV/n-1 = 3

Total Variation is 18, 
Variation because of PC1 = 15/18 = 0.8333 or 83%
Variation because of PC2 = 3/18=0.1667  or 17%

In the PC graph, the horizontal line is PC1 and in () the 83% shows variation because of PC1.

***Step 11:*** This can be expanded to n dimensions depending on the variables we have. In the end, we can choose the PC which accounts for maximum variation. Say we had 30 variables, but when we do PCA, we find 8 PC accounts for more than 95% of the data, so instead of using 30 variables, we can use 8 PCs for ML projects.
