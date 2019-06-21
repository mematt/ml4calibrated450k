#------------------------------------------------------------------------
# calibrate RF (vRF & tRF) scores with mutlinomial ridge regression (MR)
#
#     tRF tuned for >
#       - Brier score (BS),
#       - misclassification error (ME), 
#       - multiclass log loss (LL)
#
#     vRF > 
#       - vanilla (no hyperparam. tuning)
#
# Mate Maros
# maros@uni-heidelberg.de
#
# 2019-04-26 UTC 
#------------------------------------------------------------------------

calibrator_integrated_wrapper <- function(out.path = "vRF-calibrated-MR/", 
                                          out.fname = "probsCVfold",
                                          nfolds.. = NULL,
                                          y.. = NULL,
                                          load.path.w.name = "./vRF/CVfold.", # default output of the `run_nestedcv_tunedRF()` function
                                          which.optimized.metric.or.algorithm = c("brier", "miscerr", "mlogl", "vanilla", "svm", "xgboost"),
                                          save.metric.name.into.output.file = T,
                                          which.calibrator = c("Platt-LR", "Platt-FLR", "ridge-MR", "all"),
                                          verbose.messages = F,
                                          brglm.ctrl.max.iter = 10000,
                                          parallel.cv.glmnet = T,
                                          setseed = 1234){
  
    # 3. Train calibration model [on all the combined inner fold calibration sets (Figure 1., green rectangle)]: -----------------------------------------------------------------------------
    # 3.1 Platt scaling with LR --------------------------------------------------------------------------------------------------------------------------------------------------------------
    if(which.calibrator == "Platt-LR" || which.calibrator == "all") {
      calibrate_LR(out.path = out.path, out.fname = out.fname, nfolds.. = nfolds.., y.. = y.., load.path.w.name = load.path.w.name, 
                   which.optimized.metric.or.algorithm = which.optimized.metric.or.algorithm, 
                   save.metric.name.into.output.file = save.metric.name.into.output.file, 
                   verbose.messages = verbose.messages)
    }
    
    # 3.2. Platt scaling with Firth's penalized LR (FLR) ----------------------------------------------------------------------------------------------------------------------------
    if(which.calibrator == "Platt-FLR" || which.calibrator == "all") {
      calibrate_FLR(out.path = out.path, out.fname = out.fname, nfolds.. = nfolds.., y.. = y.., load.path.w.name = load.path.w.name, 
                    which.optimized.metric.or.algorithm = which.optimized.metric.or.algorithm, 
                    save.metric.name.into.output.file = save.metric.name.into.output.file, 
                    verbose.messages = verbose.messages, brglm.ctrl.max.iter = brglm.ctrl.max.iter)
    }
  
    # 3.3 Multinomial ridge regression (MR) -----------------------------------------------------------------------------------------------------------------------------------------  
    if(which.calibrator == "ridge-MR" || which.calibrator == "all") {
      calibrate_MR(out.path = out.path, 
                   out.fname = out.fname,
                   nfolds.. = nfolds.., 
                   y.. = y.., 
                   load.path.w.name = load.path.w.name, 
                   which.optimized.metric.or.algorithm = which.optimized.metric.or.algorithm, 
                   save.metric.name.into.output.file = save.metric.name.into.output.file, 
                   verbose.messages = verbose.messages, parallel.cv.glmnet = parallel.cv.glmnet,
                   setseed = setseed)
    }
}

# Function call ----------------------------------------------------------------------------------------------------
# library(glmnet)
# library(doMC)
# library(brglm)
# 
# #n_threads <- detectCores()-1
# n_threads
# registerDoMC(cores = n_threads)
# #registerDoMC(cores = 11)
# getDoParWorkers()
# getDoParVersion()
# getDoParRegistered()

# # vRF + MR ----------------------------------------------------------------------------------------------------
# # Run time 7-8mins @ 10 threads 
# Sys.time() 
# calibrate_RF_MR(out.path = "/vRF-calibrated-MR/", 
#                 out.fname = "probsCVfold",
#                 nfolds.. = NULL,
#                 y.. = NULL,
#                 load.path.w.name = "./vRF/CVfold.", 
#                 verbose.messages = T,
#                 which.optimized.metric.or.algorithm = "vanilla",
#                 parallel.cv.glmnet = T,
#                 setseed = 1234)
# Sys.time()
# 
# 
# scores.pred.rf.tuned.brier, scores.pred.rf.tuned.miscerr, scores.pred.rf.tuned.mlogl,
#
# # tRF_BS + {LR; FLR; MR} ----------------------------------------------------------------------------------------------------
# Sys.time()
# t_calib_integrated_all_tRF_BS <- system.time(
#   calibrator_integrated_wrapper(out.path = "./tRF-BS-calibrator-integrated/",
#                                 load.path.w.name = "./tRF/CVfold.",
#                                 which.optimized.metric.or.algorithm = "brier",
#                                 which.calibrator = "all",
#                                 verbose.messages = F,
#                                 save.metric.name.into.output.file = T,
#                                 parallel.cv.glmnet = T,
#                                 setseed = 1234)
#   
# )
# Sys.time()
# # # tRF_ME + {LR; FLR; MR} ----------------------------------------------------------------------------------------------------
# Sys.time()
# t_calib_integrated_all_tRF_ME <- system.time(
#   calibrator_integrated_wrapper(out.path = "./tRF-ME-calibrator-integrated/",
#                                 load.path.w.name = "./tRF/CVfold.",
#                                 which.optimized.metric.or.algorithm = "miscerr",
#                                 which.calibrator = "all",
#                                 verbose.messages = F,
#                                 save.metric.name.into.output.file = T,
#                                 parallel.cv.glmnet = T,
#                                 setseed = 1234)
#   
# )
# Sys.time()
# 
# # # tRF_LL + {LR; FLR; MR} ----------------------------------------------------------------------------------------------------
# Sys.time()
# t_calib_integrated_all_tRF_LL <- system.time(
#   calibrator_integrated_wrapper(out.path = "./tRF-LL-calibrator-integrated/",
#                                 load.path.w.name = "./tRF/CVfold.",
#                                 which.optimized.metric.or.algorithm = "mlogl",
#                                 which.calibrator = "all",
#                                 verbose.messages = F,
#                                 save.metric.name.into.output.file = T,
#                                 parallel.cv.glmnet = T,
#                                 setseed = 1234)
#   
# )
# Sys.time()

# # XGBOOST + MR ----------------------------------------------------------------------------------------------------
# Sys.time()
# t_calib_integrated_all <- system.time(
#   calibrator_integrated_wrapper(out.path = "./XGB-calibrator-integrated/", 
#                                 load.path.w.name = "./XGBOOST/CVfold.",
#                                 which.optimized.metric.or.algorithm = "xgboost",
#                                 which.calibrator = "all",
#                                 verbose.messages = T,
#                                 save.metric.name.into.output.file = F,
#                                 parallel.cv.glmnet = T,
#                                 setseed = 1234)
#   
# )
# Sys.time()
# t_calib_integrated_all
# user   system  elapsed 
# 3281.776  171.208 1181.732  # 19-20 mins @ 11 cores i9 rMBP




