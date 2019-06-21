#----------------------------------------------------------------------
# Utility functions for the Integrated Performance evaluator function 
#
#  Including all evaluated metrics:
#          - Brier score (BS)
#          - Misclassification Error (ME) 
#          - Multiclass AUC (AUC)
#          - Multiclass log loss (LL)
#
# Matt Maros & Martin Sill
# maros@uni-heidelberg.de & m.sill@dkfz.de

#
# 2019-04-16 UTC
#---------------------------------------------------------------------
 

# Brier score (BS) -------------------------------------------------------------------------------------------------------------------------------------------------------------------

brier <- function(scores,y){
  ot <- matrix(0,nrow=nrow(scores),ncol=ncol(scores)) 
  # It can cause error: 
  # encountered errors in user code, all values of the jobs will be affectedError in matrix(0, nrow = nrow(scores), ncol = ncol(scores))
  arr.ind <- cbind(1:nrow(scores),match(y,colnames(scores)))
  ot[arr.ind] <- 1
  sum((scores - ot)^2)/nrow(scores)
}



#  Multiclass log loss (LL) ----------------------------------------------------------------------------------------------------------------------------------------------------------

mlogloss <- function(scores,y){
  N <- nrow(scores)
  y_true <- matrix(0,nrow=nrow(scores),ncol=ncol(scores))
  arr.ind <- cbind(1:nrow(scores),match(y,colnames(scores)))
  y_true[arr.ind] <- 1
  eps <- 1e-15 # we use Kaggle's definition of multiclass log loss with this constrain (eps) on extremly marginal scores (see reference below)  
  scores <- pmax(pmin(scores, 1 - eps), eps)
  (-1 / N) * sum(y_true * log(scores))
}

# Reference 
# <https://web.archive.org/web/20160316134526/https://www.kaggle.com/wiki/MultiClassLogLoss>



# Misclassification Error (ME) -------------------------------------------------------------------------------------------------------------------------------------------------------

# Subfunction for misclassification error (ME)
subfunc_misclassification_rate <- function(y.true.class, y.predicted){
  error_misclass <- sum(y.true.class != y.predicted)/length(y.true.class)
  return(error_misclass)
}



# Multiclass AUC after (Hand & Till 2001) --------------------------------------------------------------------------------------------------------------------------------------------- 

# Check & install package HandTill2001 if not in namespace & load
if (!requireNamespace("HandTill2001", quietly = TRUE)) { 
  install.packages("HandTill2001")
  library(HandTill2001) } else {library(HandTill2001)}

# Subfunction for multiclass AUC & ROC by Hand & Till 2001

# Note: sum of row scores/probabilities must be scaled to 1 
subfunc_multiclass_AUC_HandTill2001 <- function(y.true.class, y.pred.matrix.rowsum.scaled1){
  auc_multiclass <- HandTill2001::auc(multcap(response = as.factor(y.true.class), 
                                              predicted = y.pred.matrix.rowsum.scaled1))
  return(auc_multiclass)
}