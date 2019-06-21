#-------------------------------------------------------------------------------------------------------------
# ml4calibrated450k - Elastic Net penalized Multinomial Logistic Regression (ELNET) - Utility/Subfunctions 
# 
#                   
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#-------------------------------------------------------------------------------------------------------------


# Check, install|load recquired packages -----------------------------------------------------------------------------------------------
if (!requireNamespace("doMC", quietly = TRUE)) {
  install.packages("doMC")
  library(doMC) } else {library(doMC)}

if (!requireNamespace("glmnet", quietly = TRUE)) { 
  install.packages("glmnet", dependencies = T)
  library(glmnet) 
} else {library(glmnet)}

if (!requireNamespace("c060", quietly = TRUE)) { 
  install.packages("c060")
  library(c060) } else {library(c060)}


## Utility functions ---------------------------------------------------------------------------------------------------------------------------

### Subfunctions 

####  Nested CV scheduler / stopper ---------------------------------------------------------------------------------------------------------------------------

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

#### Alpha-grid tuning with concurrent lambda tuning --------------------------------------------------------------------------------------------------------------------------------------------------

# Please note: in the glmnet vignette: `standardize` is a logical flag for `x` variable standardization, prior to fitting the model sequence. 
# The coefficients are always returned on the original scale. Default is `standardize=TRUE`.

##### Subfunction alpha-lambda tuner version 2 (v2) using nested parallelism with mclapply * cv.glmnet() ------------------------------------------------------------------------------------------------------------------------------------

subfunc_cvglmnet_tuner_mc_v2 <- function(x, # needs to be a matrix object # df gives error!´
                                         y, 
                                         family = "multinomial", 
                                         type.meas = "mse", 
                                         alpha.min = 0, alpha.max = 1, by = 0.1, 
                                         n.lambda = 100, 
                                         lambda.min.ratio = 10^-6, 
                                         nfolds = 10, 
                                         balanced.foldIDs = T, # use balanced foldIDs (c060 package)
                                         seed = 1234, 
                                         parallel.comp = T, 
                                         mc.cores=2L){
  # Alpha grid
  alpha.grid <- as.list(seq(alpha.min, alpha.max, by))
  
  # Fixed foldIDs are needed so that the runs are comparable between alpha settings # see help & vignette cv.glmnet
  message("Setting balanced foldIDs with nfold = ", nfolds, " @ ", Sys.time())
  set.seed(seed)
  if(balanced.foldIDs){foldIDs <- balancedFolds(class.column.factor = y, cross.outer = nfolds)} 
  # c060::balancedFolds() => "::" sometimes caused crash with parallel
  # {foldIDs <- sample(1:nfolds, size = length(y), replace = TRUE)} 
  
  # Define new version/repurposed `cv.glmnet()` function parellized for the alpha grid 
  parallel.cvglmnet <- function(i){
    message("Tuning cv.glmnet with alpha = ", i, " @ ", Sys.time())
    set.seed(seed+1, kind ="L'Ecuyer-CMRG")
    # Inner loop of nlamba CV also using foreach %dopar% within cv.glmnet
    cv.glmnet(x = x, 
              y = y, 
              alpha = i, 
              family = family, 
              type.measure = type.meas,                 
              nlambda = n.lambda, 
              lambda.min.ratio = lambda.min.ratio,
              foldid = foldIDs,
              parallel = parallel.comp) # still allow nested parallelism (foreach) for CV of nlambda
  }
  
  # Outerloop using mclapply on the alpha[[i]] and runs parallel.cv.glmnet() function that is also parallelized with foreach %dopar% for 10fold lambda CV
  # Hence the name "crazy" nested parallelism ==> this was the fastest implementation; 
  # See also blog post by Max Kuhn: <http://appliedpredictivemodeling.com/blog/2018/1/17/parallel-processing> ; <https://topepo.github.io/caret/parallel-processing.html>
  message("Start mclapply ", " @ ", Sys.time())
  cvfit.l <- mclapply(alpha.grid, parallel.cvglmnet, 
                      mc.preschedule = T, mc.set.seed = T, mc.cores = mc.cores)  
  
  
  return(cvfit.l)                                                                
}


#### Extract alpha and lambda_1se values with lowest cv.error -----------------------------------------------------------------------------------------------------------------------------------------

subfunc_glmnet_mse_min_alpha_extractor <- function(resl, # resl is the output list of length (alpha.grid) #  generated by subfunc_cvglmnet_tuner_mc_v2()
                                                   lambda.1se = T, # if lambda.1se = F => lambda.min will be used
                                                   alpha.min = 0, 
                                                   alpha.max = 1){  
  outl <- list(length(resl))
  if(lambda.1se){
    outl <- sapply(seq_along(resl), function(i){ 
      resl[[i]]$cvm[resl[[i]]$lambda == resl[[i]]$lambda.1se] 
      # $cvm is the cross validated measure in our case MSE 
      # gets the smallest $cvm @ the lambda.1se location
    }) 
  } else {
    outl <- sapply(seq_along(resl), function(i){
      resl[[i]]$cvm[resl[[i]]$lambda == resl[[i]]$lambda.min]
    })
  }
  # ID of smallest $cvm measure ("mse") @lambda.1se accross all CV alphas-lambda model pairs
  l <- which.min(outl)          # which.min only operates on vectors => sapply() 
  # Get alpha value with above ID
  alphas <- seq(alpha.min, alpha.max, length.out = length(resl)) # assuming equal length/by division within alpha range min-max 0-1 => 0.1 => 11
  opt.alpha <- alphas[[l]]
  
  # Extracts optimal lambda either .1se (suggested by GLMNET vignette more robust estimate see. Breiman 1984) or at lambda.min
  if(lambda.1se){opt.lambda <- resl[[l]]$lambda.1se}
  else{opt.lambda <- resl[[l]]$lambda.min}
  
  # Gets directly the ID (l) with min $cvm => needed for predict()
  opt.model <- resl[[which.min(outl)]] # same as resl[[l]]
  
  # Results
  res <- list(cvm.alpha.i.list = outl, 
              opt.id = l, 
              opt.alpha = opt.alpha, 
              opt.lambda = opt.lambda, 
              opt.mod = opt.model) # model object `glmnet.fit` is => ext.cvfit.v2$opt.mod$glmnet.fit
  return(res)
}
