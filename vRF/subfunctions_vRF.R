#--------------------------------------------------------------------
# ml4calibrated450k - vanilla RF (vRF) - Utility/Subfunctions 
#
#
# Martin Sill & Matt Maros
# m.sill@dkfz.de & maros@uni-heidelberg.de
#
# 2019-04-18 
#--------------------------------------------------------------------


## Load required libraries, data objects 

# Check, install | load recquired packages ---------------------------------------------------------------------------------------------------------------------------

if (!requireNamespace("randomForest", quietly = TRUE)) { 
  install.packages("randomForest")
  library(randomForest) } else {library(randomForest)}

if (!requireNamespace("doMC", quietly = TRUE)) { 
  install.packages("doMC")
  library(doMC) } else {library(doMC)}


## Utility functions ---------------------------------------------------------------------------------------------------------------------------

### Subfunctions 

############################################
### 0. Nested CV scheduler / stopper     ###
############################################

subfunc_nestedcv_scheduler <- function(K, K.start, K.stop, k.start, k.stop, n.cv.folds, n.cv.inner.folds){
  # Correctly stop nested CV
  if(K > K.start && K < K.stop) {
    k.start <- 0
    n.cv.inner.folds <- n.cv.folds 
    
  } else {
    if(K == K.start) {
      if(K.start != K.stop) { # && k.start != 0){
        n.cv.inner.folds <- n.cv.folds # k inner goes to .5
      } else { # K == K.start == K.stop && k.start == 0  
        n.cv.inner.folds <- k.stop
      }
    } else { # K == K.stop
      if(k.start != 0) k.start <- 0
      n.cv.inner.folds <- k.stop
    } 
  }
  res <- list(k.start = k.start, 
              n.cv.inner.folds = n.cv.inner.folds
  )
  return(res)
}

############################################
### 1. Parallelized randomForest wrapper ###
############################################

rfp <- function(xx, ..., ntree = ntree, mc = mc, seed = NULL){
  if(!is.null(seed)) set.seed(seed, "L'Ecuyer") # help("mclapply") # reproducibility
  rfwrap <- function(ntree, xx, ...) randomForest::randomForest(x = xx, ntree = ntree, norm.votes = FALSE, ...)
  rfpar <- mclapply(rep(ceiling(ntree / mc), mc), mc.cores = mc, rfwrap, xx = xx, ...) 
  do.call(randomForest::combine, rfpar) # > ?randomForest::combine => combines two or more trees/ensembles of trees (rF objects) into one # get the subset of trees from each core and patch them back together
}