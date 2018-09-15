![alt text](https://github.com/ggeop/Bayes-Naives-Classifier-ML/blob/master/imgs/cover.png)
(cover image from: https://chrisalbon.com/) :-)

# Bayes-Naives-Classifier
## Description
We have created a very simple data set consisting of ten observations of seven input features. This is a simplified rendition of the German Credit data set from the UCI repository for the Support Vector Machine lab. The output variable is the Decision column. Here, this takes the value 1 when we reject the loan and 0 when we accept. We are going to construct a very simple Naïve Bayes model for this problem but we are going to train this manually.

## Implementation

Firstly, we create a small dataset in order to test our code

```
d1 <- c(1, 0, 0, 1, 0, 0, 0, 1)
d2 <- c(0, 1, 0, 1, 1, 0, 0, 0)
d3 <- c(0, 0, 1, 0, 0, 0, 1, 0)
d4 <- c(0, 0, 0, 1, 0, 0, 0, 0)
d5 <- c(0, 0, 0, 0, 0, 0, 1, 0)
d6 <- c(0, 0, 0, 1, 0, 1, 0, 1)
d7 <- c(0, 0, 1, 0, 0, 1, 0, 1)
d8 <- c(1, 0, 0, 0, 0, 0, 0, 1)
d9 <- c(0, 0, 0, 0, 0, 1, 0, 1)
d10 <- c(1, 1, 0, 0, 0, 1, 0, 1)
nb_df <- as.data.frame(rbind(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10))
names(nb_df) <- c("BadCredit", "HasStableJob", "OwnsHouse", "BigLoan",
"HasLargeBankAccount", "HasPriorLoans", "HasDependents", "Decision")
```
Then we have to build piece by piece the Naive Bayes Classifier

```
#Create the Class Vector
decision<-nb_df$Decision

#Calculate the propability of the loan accept
p_accept<-sum(decision==0)/length(decision)

#calculate the propability of the loan rejection
p_reject<-sum(decision==1)/length(decision)

#Create a vector with the prior probabilities
priors<-c(p_accept,p_reject)

```
Compute a summary data frame in which one row contains the probabilities P(Fi = 1|Class = 0) for all the different features Fi and the
other row contains the probabilities P(Fi = 1|Class = 1). For example, the cell at [1,1] could contain the probability that when we accept the loan (class = 0) the loan applicant has bad credit (BadCredit = 1).

```

aggregate(x=nb_df[c("BadCredit","HasStableJob","OwnsHouse","BigLoan","HasLargeBankAccount","HasPriorLoans","HasDependents")],
          by =nb_df[c("Decision")],
          FUN = function(x){y <- sum(x)/length(x); return(y)}
                  )
                  
aggregate(x=nb_df[c("BadCredit","HasStableJob","OwnsHouse","BigLoan","HasLargeBankAccount","HasPriorLoans","HasDependents")],
          by =nb_df[c("Decision")],
          FUN = function(x){y <- 1- sum(x)/length(x); return(y)}
                  )

```

Recalculate the matrix of probabilities that we computed in previous step to incorporate additive smoothing.

```
#Calculate again the propabilities
prob_matrix<-aggregate(x=nb_df[c("BadCredit","HasStableJob","OwnsHouse","BigLoan","HasLargeBankAccount","HasPriorLoans","HasDependents")],
          by =nb_df[c("Decision")],
          FUN = function(x){y <- (sum(x)+1)/(length(x)+2); return(y)}
                  )


```

Finaally, we create the Bayes Naive classifier

```
classifier<-function(observation,priors, prob_matrix)
  {
    #Delete the Decision column
    observation$Decision<-NULL
    prob_matrix$Decision<-NULL
    
    #Culculate the probability for the Reject Class (C=1)
    p<-c()
    for (i in 1:ncol(observation))
    {
      if (observation[i]==1)
      {
        p[i]<-prob_matrix[2,i]
      }
      else
      {
        p[i]<-1-prob_matrix[2,i]
      }
    } 
    
    prob_reject<-prod(p)*priors[2]
    
    #Culculate the probability for the Accept Class (C=0)
    p<-c()
    for (i in 1:ncol(observation))
    {
      if (observation[i]==1)
      {
        p[i]<-prob_matrix[1,i]
      }
      else
      {
        p[i]<-1-prob_matrix[1,i]
      }
    } 
    
    prob_accept<-prod(p)*priors[1]
   
   #Assign to the highest probability 
    if(prob_accept>prob_reject)
    {
      return(0) 
    }
    else
    {
      return(1)
    }
    
    
}

predict_nb <- function(test_df, priors, prob_matrix) 
{

  predict<-c()
  for (i in 1:nrow(test_df))
  {
    predict[i]<-classifier(test_df[i,],priors, prob_matrix)
  }


  return(predict)
}


```

Compute the training accuracy of your Naïve Bayes model using the function that you just created

```

#Creare the accuracy function
accuracy<-function(test_dataset, predict_values)
{
  count<-0
  for (i in 1:length(test$Decision))
  {
    if (prediction[i] == test$Decision[i])
    {
      count<-count+1
    }
  }
  return(count/length(test$Decision))
}


#We know that we don't have a test dataset, so we take a partition of the original dataset

test<-nb_df[1:3,]
prediction<-predict_nb(test, priors, prob_matrix)

#We calculate the accuracy. It's normal to have accuracy equals to 1 because we run it in the same dataset.

#Calculate the accuracy
accuracy(test,prediction)

```
