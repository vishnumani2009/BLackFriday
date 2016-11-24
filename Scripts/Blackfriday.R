
## setting working directory
path <- "D:/BLackFriday/data"   # edit the path where the data is located
setwd(path)


## loading libraries
library(dummies)
library(plyr)
library(xgboost)


## function for importing Rscript from github
source_https <- function(url)
{
  library(RCurl)
  eval(parse(text=getURL(url,followlocation=T,cainfo=system.file("CurlSSL","cacert.pem",package="RCurl"))),envir=.GlobalEnv)
}

XGBoost <- function(X_train,y,X_test=data.frame(),cv=5,transform="none",objective="binary:logistic",eta=0.1,max.depth=5,nrounds=50,gamma=0,min_child_weight=1,subsample=1,colsample_bytree=1,seed=123,metric="auc",importance=0)
{
  # defining evaluation metric
  score <- function(a,b,metric)
  {
    switch(metric,
           accuracy = sum(abs(a-b)<=0.5)/length(a),
           auc = auc(a,b),
           logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),
           mae = sum(abs(a-b))/length(a),
           precision = length(a[a==b])/length(a),
           rmse = sqrt(sum((a-b)^2)/length(a)),
           rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))           
  }
  
  if (metric == "auc")
  {
    library(pROC)
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.numeric(y)
  
  if (transform == "log")
  {
    X_train$result <- log(X_train$result)
  }
  
  # converting data to numeric
  for (i in 1:ncol(X_train))
  {
    X_train[,i] <- as.numeric(X_train[,i])
  }
  
  if (nrow(X_test)>0)
  {
    for (i in 1:ncol(X_test))
    {
      X_test[,i] <- as.numeric(X_test[,i])
    }    
  }
  
  X_train[is.na(X_train)] <- -1
  X_test[is.na(X_test)] <- -1
  
  X_test2 <- X_test
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  # cross-validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i)
    X_val <- subset(X_train, randomCV == i)
    
    feature_names <- colnames(subset(X_build, select = -c(order, randomCV, result)))
    
    build <- as.matrix(subset(X_build, select = -c(order, randomCV, result)))
    val <- as.matrix(subset(X_val, select = -c(order, randomCV, result)))
    test <- as.matrix(X_test2)
    
    build_label <- as.matrix(subset(X_build, select = c('result')))
    
    # building model
    model_xgb <- xgboost(build,build_label,objective=objective,eta=eta,max.depth=max.depth,nrounds=nrounds,gamma=gamma,min_child_weight=min_child_weight,subsample=subsample,colsample_bytree=colsample_bytree,nthread=-1,verbose=0,eval.metric="auc")
    
    # variable importance
    if (importance == 1)
    {
      print (xgb.importance(feature_names=feature_names, model=model_xgb))
    }
    
    # predicting on validation data
    pred_xgb <- predict(model_xgb, val)
    if (transform == "log")
    {
      pred_xgb <- exp(pred_xgb)
    }
    
    X_val <- cbind(X_val, pred_xgb)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_xgb <- predict(model_xgb, test)
      if (transform == "log")
      {
        pred_xgb <- exp(pred_xgb)
      }
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_xgb, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_xgb)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_xgb <- (X_test$pred_xgb * (i-1) + pred_xgb)/i
      }            
    }
    
    gc()
  } 
  
  # final evaluation score
  output <- output[order(output$order),]
  cat("\nXGBoost ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_xgb, metric), "\n", sep = "")
  
  output <- subset(output, select = c("order", "pred_xgb"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}

## loading data
train <- read.csv("./train.csv", stringsAsFactors=F)
test <- read.csv("./test.csv", stringsAsFactors=F)


## cleaning data

# removing categories 19 and 20
X_train <- subset(train, !Product_Category_1 %in% c(19,20))
X_test <- test

# onehot-encoding city variable
X_train <- dummy.data.frame(X_train, names=c("City_Category"), sep="_")
X_test <- dummy.data.frame(X_test, names=c("City_Category"), sep="_")

# converting age variable to numeric
X_train$Age[X_train$Age == "0-17"] <- "15"
X_train$Age[X_train$Age == "18-25"] <- "21"
X_train$Age[X_train$Age == "26-35"] <- "30"
X_train$Age[X_train$Age == "36-45"] <- "40"
X_train$Age[X_train$Age == "46-50"] <- "48"
X_train$Age[X_train$Age == "51-55"] <- "53"
X_train$Age[X_train$Age == "55+"] <- "60"

X_test$Age[X_test$Age == "0-17"] <- "15"
X_test$Age[X_test$Age == "18-25"] <- "21"
X_test$Age[X_test$Age == "26-35"] <- "30"
X_test$Age[X_test$Age == "36-45"] <- "40"
X_test$Age[X_test$Age == "46-50"] <- "48"
X_test$Age[X_test$Age == "51-55"] <- "53"
X_test$Age[X_test$Age == "55+"] <- "60"

X_train$Age <- as.integer(X_train$Age)
X_test$Age <- as.integer(X_test$Age)

# converting stay in current city to numeric
X_train$Stay_In_Current_City_Years[X_train$Stay_In_Current_City_Years == "4+"] <- "4"
X_test$Stay_In_Current_City_Years[X_test$Stay_In_Current_City_Years == "4+"] <- "4"

X_train$Stay_In_Current_City_Years <- as.integer(X_train$Stay_In_Current_City_Years)
X_test$Stay_In_Current_City_Years <- as.integer(X_test$Stay_In_Current_City_Years)

# converting gender to binary
X_train$Gender <- ifelse(X_train$Gender == "F", 1, 0)
X_test$Gender <- ifelse(X_test$Gender == "F", 1, 0)

# feature representing the count of each user
user_count <- ddply(X_train, .(User_ID), nrow)
names(user_count)[2] <- "User_Count"
X_train <- merge(X_train, user_count, by="User_ID")
X_test <- merge(X_test, user_count, all.x=T, by="User_ID")

# feature representing the count of each product
product_count <- ddply(X_train, .(Product_ID), nrow)
names(product_count)[2] <- "Product_Count"
X_train <- merge(X_train, product_count, by="Product_ID")
X_test <- merge(X_test, product_count, all.x=T, by="Product_ID")
X_test$Product_Count[is.na(X_test$Product_Count)] <- 0

# feature representing the average Purchase of each product
product_mean <- ddply(X_train, .(Product_ID), summarize, Product_Mean=mean(Purchase))
X_train <- merge(X_train, product_mean, by="Product_ID")
X_test <- merge(X_test, product_mean, all.x=T, by="Product_ID")
X_test$Product_Mean[is.na(X_test$Product_Mean)] <- mean(X_train$Purchase)

# feature representing the proportion of times the user purchases the product more than the product's average
X_train$flag_high <- ifelse(X_train$Purchase > X_train$Product_Mean,1,0)
user_high <- ddply(X_train, .(User_ID), summarize, User_High=mean(flag_high))
X_train <- merge(X_train, user_high, by="User_ID")
X_test <- merge(X_test, user_high, by="User_ID")

# subsetting columns for submission
submit <- X_test[,c("User_ID","Product_ID")]

# target variable
y <- X_train$Purchase

# removing irrelevant columns
X_train <- subset(X_train, select=-c(Purchase,Product_ID,flag_high))
X_test <- subset(X_test, select=c(colnames(X_train)))


## xgboost with cross validation
#source_https("https://github.com/vasanthgx/Models_CV/blob/master/XGBoost.R")
model_xgb_1 <- XGBoost(X_train,y,X_test,cv=5,objective="reg:linear",nrounds=500,max.depth=10,eta=0.1,colsample_bytree=0.5,seed=235,metric="rmse",importance=1)


## submission file
test_xgb_1 <- model_xgb_1[[2]]

# adding predictions
submit$Purchase <- test_xgb_1$pred_xgb

# tweaking final predictions (You know, to get those extra decimals :-) )
submit$Purchase[submit$Purchase < 185] <- 185
submit$Purchase[submit$Purchase > 23961] <- 23961

write.csv(submit, "./submit.csv", row.names=F)

## model performance (RMSE)

# CV = 2425.38
# Public LB = 2428.51
# Private LB = 2434.13

