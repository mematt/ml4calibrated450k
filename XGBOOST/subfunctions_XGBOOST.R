#--------------------------------------------------------------------
# ml4calibrated450k - XGBOOST - Utility/Subfunctions 
#
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-28 
#--------------------------------------------------------------------

## Load required libraries, data objects 

# Check, install | load recquired packages ---------------------------------------------------------------------------------------------------------------------------
if (!requireNamespace("caret", quietly = TRUE)) { 
  install.packages("caret", dependencies = T)
  library(caret) 
} else {library(caret)}


if (!requireNamespace("xgboost", quietly = TRUE)) { 
  install.packages("xgboost")
  library(xgboost) } else {library(xgboost)}

if (!requireNamespace("Matrix", quietly = TRUE)) { 
  install.packages("Matrix")
  library(Matrix) } else {library(Matrix)}

if (!requireNamespace("doMC", quietly = TRUE)) {
  install.packages("doMC")
  library(doMC) } else {library(doMC)}

# <CRITICAL> Please note that on Mac OSX only a single-threaded version of `xgboost` will be installed when using the install.packages(“xgboost”) command. 
# This is because the default Apple Clang compiler does not support OpenMP. 
# To enable multi-threading on Mac OSX please consult the xgboost installation guide 
# (https://xgboost.readthedocs.io/en/latest/build.html#osx-multithread).


## Utility functions ---------------------------------------------------------------------------------------------------------------------------

# <NOTE> The XGboost workflow does not have additional "real" utility/subfunctions everything is integrated within the `trainXGBOOST_caret_tuner()`. 

### Subfunction 

#### Nested CV scheduler/stopper

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