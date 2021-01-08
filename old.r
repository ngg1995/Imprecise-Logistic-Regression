library(MASS)    
library(caTools)     # AUC calculation
library(intervals)   # for plotting intervals
library(ROCR)        # ROC plots 

#setwd("~/Documents/AB/Intervalized_Logistic_Regression/")

# clear vars
rm(list = ls())

# Global plotting
doPlots = FALSE

####################################
# Data 
####################################

# Source: http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data

# Make sure this R file and SAheart.data are in the same directory.
hr = read.table("SAheart.data", sep=",",head=T,row.names=1)

# Pair-wise plots
pairs(hr[1:9], pch=21, bg=c("red","green")[factor(hr$chd)])

# Build train and test sets
set.seed(777) 
n = nrow(hr)
p = ncol(hr)-1
test.ratio = 0.2  # proportion of data to use for test set
n.test = ceiling(n*test.ratio)
idxTest = sample(1:n,n.test) # ramdomly select the indices for the test set
idxTrain = setdiff(1:n,idxTest) # the training set is everything not in the test set

data.train = hr[idxTrain,] # train set
data.test = hr[idxTest,]   # test set

print(paste("Trainset length: ", toString(length(idxTrain))))
print(paste("Testset length: ", toString(length(idxTest))))
print(paste("Total prevalence: ", toString(length(which(hr$chd == 1)) / length(hr$chd))))
print(paste("Trainset prevalence: ", toString(length(which(data.train$chd == 1)) / length(data.train$chd))))
print(paste("Testset prevalence: ", toString(length(which(data.test$chd == 1)) / length(data.test$chd))))

####################################
# Logistic regression
####################################

# Fit model on the training set with all predictors
hr.lr.full = glm(chd ~ sbp + tobacco + ldl + famhist + obesity + alcohol + age, family=binomial(link="logit"), data=data.train)

summary(hr.lr.full)

# Compute confidence interval on coefficients
confint(hr.lr.full, level=0.95) 

# Predict probabilities for the test set
ypredFull = predict(hr.lr.full, data.test, type="response")

# Compute the ROC curve with Area Under Curve (AUC)
aucFull = colAUC(ypredFull, data.test$chd, plotROC=TRUE)

# Remove factors one-by-one using AIC
hr.lr.small = stepAIC(hr.lr.full)
summary(hr.lr.small)

# When interpreting the coefficients, remember we're dealing with log odds. Thus the exp(),
exp(hr.lr.small$coefficients)

ypredSmall = predict(hr.lr.small, data.test, type="response") 
aucSmall = colAUC(ypredSmall, data.test$chd, plotROC=doPlots)

aucFull
aucSmall

####################################
# Intervalize the Data 
####################################

# In the smaller model above, we have risk factors for tabacco, LDL, family history, and age. We will assume interval data for tabacco and LDL since family history is binary and age is usually known. 

hist(hr$tobacco, main='Tobacco')
hist(hr$ldl, main='LDL')

# Function to embed measurements (all of which are > 0) within a Gaussian error interval
# The observed data point is put somewhere within the interval.
interfy = function(data, sigmaPct) {
   if (any(data < 0) == TRUE) {
   	   print("All data should be > 0")
   		return(NULL)
	}
	n = length(data)
	dSigma = sd(data)
	eSigma = dSigma * sigmaPct

	zeros = matrix(0,n,1)
	errorLeft  = rnorm(n, mean=0, sd=eSigma)
    errorRight = rnorm(n, mean=0, sd=eSigma)
   
	left  = pmax(zeros, data - abs(errorLeft))
	right = data + abs(errorRight)
	
	if (any(left > right) == TRUE) {
	   print("Left > right")
		return(NULL)
	}
   interval = NULL
   interval$sigmaPct = sigmaPct
   interval$dSigma = dSigma
   interval$eSigma = eSigma
   interval$width = right-left
   interval$left = left
   interval$right = right
   return(interval)
}

# Function to plot intervals
plotInt = function(interval, title) {
	plot(Intervals(cbind(interval$left,interval$right)), main=title)
}

# copy the orignal data structure
hri = hr

# Percent of data standard dev to use as interval width.
sigmaPct = .1

# Intervalize Tobacco
intTob = interfy(hr$tobacco, sigmaPct)
hri$tobaccoL = intTob$left
hri$tobaccoR = intTob$right
plotInt(intTob, 'Tobacco Intervals')

# Intervalize LDL
intLDL = interfy(hr$ldl, sigmaPct)
hri$ldlL = intLDL$left
hri$ldlR = intLDL$right
plotInt(intLDL, 'LDL Intervals')

# Add midpoints of LDL and tobaccco intervals to main dataset
hri$ldlM = (hri$ldlL + hri$ldlR)/2
hri$tobaccoM = (hri$tobaccoL + hri$tobaccoR)/2

# Drop the old predictors
hri$tobacco = NULL
hri$ldl = NULL

# Define the intervalized train/test sets, using the same indices for samples.
intData.train = hri[idxTrain,] # train set
intData.test = hri[idxTest,]   # test set

####################################
# Type 1 Intervalized Regression
####################################

# Use both endpoints in a single logistic regression fit.
hri.lr.type1 = glm(chd ~ tobaccoL + tobaccoR + ldlL + ldlR + famhist + age, family=binomial(link="logit") , data=intData.train)

ypred1 = predict(hri.lr.type1, intData.test, type="response") 
aucType1 = colAUC(ypred1, intData.test$chd, plotROC=doPlots)
aucType1

####################################
# Type 2 Intervalized Regression
####################################

# Fit two models for each endpoint.
hri.lr.type2.left  = glm(chd ~ tobaccoL + ldlL + famhist + age, family=binomial(link="logit") , data=intData.train) 

hri.lr.type2.right = glm(chd ~ tobaccoR + ldlR + famhist + age, family=binomial(link="logit") , data=intData.train) 

ypredL = predict(hri.lr.type2.left, intData.test, type="response")
ypredR = predict(hri.lr.type2.right, intData.test, type="response") 
ypred2 = (ypredL + ypredR)/2 

aucType2 = colAUC(ypred2, intData.test$chd, plotROC=doPlots)
aucType2

####################################
# Type M Intervalized Regression
####################################

# Using the midpoints of the interval data
hri.lr.typeM = glm(chd ~ tobaccoM + ldlM + famhist + age, family=binomial(link="logit") , data=intData.train) 

ypredM = predict(hri.lr.typeM, intData.test, type="response") 
aucTypeM = colAUC(ypredM, intData.test$chd, plotROC=doPlots)
aucTypeM

####################################
# Compare ROCs
####################################

perf1 = performance(prediction(ypred1, intData.test$chd),"tpr","fpr")  # Type 1
perf2 = performance(prediction(ypred2, intData.test$chd),"tpr","fpr")  # Type 2
perfM = performance(prediction(ypredM, intData.test$chd),"tpr","fpr")  # Type M
perfS = performance(prediction(ypredSmall, data.test$chd),"tpr","fpr") # Original, non-intervalized

plot(perf1, col='green', main='ROCs')
par(new=TRUE)
plot(perf2, col='blue')
par(new=TRUE)
plot(perfM, col='cyan')
par(new=TRUE)
plot(perfS, col='red')

aucSmall
aucType1
aucType2
aucTypeM
