#-------------------------------------------------------------------------------------------------------------
# ml4calibrated450k - Elastic Net penalized Multinomial Logistic Regression (ELNET) - Train function
# 
#                   
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#-------------------------------------------------------------------------------------------------------------

### Training & tuning function - integrating the utility/subfuctions from `subfunctions_GLMNET.R`

trainGLMNET <- function(y, 
                        betas, 
                        seed = 1234, 
                        nfolds.cvglmnet = 10, 
                        mc.cores = 2L,
                        parallel.cvglmnet = T,
                        alpha.min = 0, alpha.max = 1, by = 0.1){
  
  ## 1. Train RF for variable selection
  set.seed(seed,kind = "L'Ecuyer-CMRG") 
  message("seed: ", seed)
  message("n: ", nrow(betas))  # n_patients
  message("cores: ", mc.cores)
  #getOption("mc.cores", mc.cores)#paste0(mc.cores, "L")) # getOption("mc.cores", detectCores()-2L)
  
  message("Start (concurrent) tuning of cv.glmnet hyperparameters alpha and lambda @  ", Sys.time())
  t1 <- system.time(
    cvfit.glmnet.tuning <- subfunc_cvglmnet_tuner_mc_v2(x = betas, 
                                                        y = y, 
                                                        seed = seed, 
                                                        nfolds = nfolds.cvglmnet,
                                                        family = "multinomial", 
                                                        type.meas = "mse", 
                                                        alpha.min = alpha.min, alpha.max = alpha.max, by = by, 
                                                        n.lambda = 100, 
                                                        lambda.min.ratio = 10^-6,
                                                        balanced.foldIDs = T,
                                                        mc.cores = mc.cores, # mc.cores is for the mclapply() shuffling through the alpha grid deafult 0-1 (x11)
                                                        parallel.comp = parallel.cvglmnet)   # argument for cv.glmnet it uses foreach() if parallel = T
  )
  
  # Extract permutation based importance measure # USE Version 2 of extractor!
  res.cvfit.glmnet.tuned <- subfunc_glmnet_mse_min_alpha_extractor(cvfit.glmnet.tuning, alpha.min = alpha.min, alpha.max = alpha.max)
  
  # Output hyperparameter results in the console
  message("Hyperparameter Tuning Results:", 
          "\n Optimal alpha: ", res.cvfit.glmnet.tuned$opt.alpha,
          "\n Optimal lambda: ", res.cvfit.glmnet.tuned$opt.lambda)
  
  message("Re-fitting optimal/tuned model on data @ ", Sys.time())
  t2 <-  system.time(
    probs.glmnet.tuned <- predict(object = res.cvfit.glmnet.tuned$opt.mod, 
                                  newx = betas, 
                                  s = res.cvfit.glmnet.tuned$opt.mod$lambda.1se, 
                                  type = "response")[,,1] # predict.glmnet() generates an array => [,,1] 
  ) 
  
  # Results
  res <- list(res.cvfit.glmnet.tuned$opt.mod, 
              probs.glmnet.tuned,  
              res.cvfit.glmnet.tuned, 
              t1, t2) 
  return(res)
}
