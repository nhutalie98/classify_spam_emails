library(rpart)
library(rpart.plot)
library(dplyr)
library(caret)
library(randomForest)
library(neuralnet)
library(nnet)
library(moments)



spambase <- read.csv("spambase.csv")

#Rename Variables whose name was changed when loading data 
spambase <- rename(spambase,"C;" = C., "C("= C..1, "C["=C..2,
                   "C!"=C..3, "C$"=C..4, "C#"= C..5)

#partition the data
set.seed(1) #ensure we get the same partitions
train.index <- sample(nrow(spambase), nrow(spambase) * 0.6)
valid.index <- as.numeric(setdiff(rownames(spambase), train.index))
spambase.train <- spambase[train.index, ]
spambase.valid <- spambase[valid.index, ]

#full classification tree
spambase.ct0 <- rpart(Spam ~ ., data = spambase.train, method = "class", cp = 0, minsplit = 1)
spambase.ct0
prp(spambase.ct0, digits = 4, type = 1, extra = 1, varlen = -10,
    box.col = ifelse(spambase.ct0$frame$var == "<leaf>", 'gray', 'white'))
#in sample prediction
pred0 <- predict(spambase.ct0, type = "class")
confusionMatrix(pred0, as.factor(spambase.train$Spam), positive = "1")
#out of sample prediction
pred1 <- predict(spambase.ct0, spambase.valid, type = "class")
confusionMatrix(pred1, as.factor(spambase.valid$Spam), positive = "1")

#pruning and finding the best-pruned tree
#cross validation
cv.ct <- rpart(Spam ~ ., data = spambase, method="class",
               cp = .00001, minsplit = 5, xval = 10)
printcp(cv.ct) 

#minimum xerror is 0.19526 at cp = 0.00082736 
#add xstd 0.19526 +  0.0099705 = 0.2052305
#choose the simplest with xerror < 0.2052305
#choose cp =  0.00137893  (42 splits)

#prune by lower cp
pruned.ct <- prune(cv.ct, cp =  0.00137893)
prp(pruned.ct, digits = 4, type = 1, extra = 1, varlen = -10,
    box.col = ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))
#out of sample prediction
pruned.pred1 <- predict(pruned.ct, spambase.valid, type = "class")
confusionMatrix(pruned.pred1, as.factor(spambase.valid$Spam), positive = "1")

#view classification rules
pruned.ct

#reload and partition the data for random forest 
spambase <- read.csv("spambase.csv")
set.seed(1) #ensure we get the same partitions
train.index <- sample(nrow(spambase), nrow(spambase) * 0.6)
valid.index <- as.numeric(setdiff(rownames(spambase), train.index))
spambase.train <- spambase[train.index, ]
spambase.valid <- spambase[valid.index, ]

#random forest
rf <- randomForest(as.factor(Spam) ~ ., data = spambase.train, ntree = 500,
                   mtry = 4, nodesize = 5, importance = TRUE)
#variable importance plot
varImpPlot(rf, type=1)
# C..3 is C! - most important
#followed by CAP_avg, C..4 (C$), and remove
#confusion matrix
rf.pred <- predict(rf, spambase.valid)
confusionMatrix(rf.pred, as.factor(spambase.valid$Spam), positive = "1")

#preprocessing for neural nets
skewness(spambase) #all predictors are highly skewed
t(t(names(spambase)))

#log transform
cols <- colnames(spambase[,c(1:57)])
for (i in cols) {
  spambase[[i]] <- log(spambase[[i]] + 1)
}
summary(spambase)

#standardize on [0,1] scale
cols <- colnames(spambase[,-58])
for (i in cols) {
  spambase[[i]] <- (spambase[[i]] - min(spambase[[i]])) / (max(spambase[[i]]) - min(spambase[[i]]))
}
summary(spambase) #everything is now on [0,1] scale

#convert spam to a factor
spambase$Spam <- as.factor(spambase$Spam)

#partition data again
set.seed(1) #ensure we get the same partitions
train.index <- sample(nrow(spambase), nrow(spambase) * 0.6)
valid.index <- as.numeric(setdiff(rownames(spambase), train.index))
spambase.train <- spambase[train.index, ]
spambase.valid <- spambase[valid.index, ]

#neural net with 1 hidden layer containing 3 nodes
nn <- neuralnet(Spam ~ ., data=spambase.train, linear.output = FALSE, hidden = 3)
#display weights
nn$weights
#display predictions
prediction(nn)
#plot network
plot(nn, rep="best")
#in sample performance
training.prediction <- compute(nn, spambase.train[, -58])
training.class <- apply(training.prediction$net.result, 1, which.max) - 1
confusionMatrix(as.factor(training.class), as.factor(spambase[train.index, ]$Spam))
#out of sample performance
validation.prediction <- compute(nn, spambase.valid[, -58])
validation.class <- apply(validation.prediction$net.result, 1, which.max) - 1
confusionMatrix(as.factor(validation.class), as.factor(spambase[valid.index, ]$Spam))

#neural net with 1 hidden layer containing 28 nodes
nn2 <- neuralnet(Spam ~ ., data=spambase.train, linear.output = FALSE, hidden = 28)
#display weights
nn2$weights
#display predictions
prediction(nn2)
#plot network
plot(nn2, rep="best")
#in sample performance
training.prediction <- compute(nn2, spambase.train[, -58])
training.class <- apply(training.prediction$net.result, 1, which.max) - 1
confusionMatrix(as.factor(training.class), as.factor(spambase[train.index, ]$Spam))
#out of sample performance
validation.prediction <- compute(nn2, spambase.valid[, -58])
validation.class <- apply(validation.prediction$net.result, 1, which.max) - 1
confusionMatrix(as.factor(validation.class), as.factor(spambase[valid.index, ]$Spam))

#neural net with 2 hidden layers containing 12 nodes each
nn3 <- neuralnet(Spam ~ ., data=spambase.train, linear.output = FALSE, hidden = c(12,12))
#display weights
nn3$weights
#display predictions
prediction(nn3)
#plot network
plot(nn3, rep="best")
#in sample performance
training.prediction <- compute(nn3, spambase.train[, -58])
training.class <- apply(training.prediction$net.result, 1, which.max) - 1
confusionMatrix(as.factor(training.class), as.factor(spambase[train.index, ]$Spam))
#out of sample performance
validation.prediction <- compute(nn3, spambase.valid[, -58])
validation.class <- apply(validation.prediction$net.result, 1, which.max) - 1
confusionMatrix(as.factor(validation.class), as.factor(spambase[valid.index, ]$Spam))
