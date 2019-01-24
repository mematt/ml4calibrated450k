# Evaluation - Log-Loss
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
# https://web.archive.org/web/20160316134526/https://www.kaggle.com/wiki/MultiClassLogLoss