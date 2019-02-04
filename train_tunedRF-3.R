
# Source required R scripts
source("subfunctions_tunedRF.R")
source("misclassification error.R")
source("mAUC.R")
source("brier.R")
source("mlogloss.R")
# Import packages
library(caret)
library(randomForest)



############################################################
###     RF HYPER PARAMETER TUNING                        ###
###         using CARET customRF() FUNCTION              ###
###                                                      ###
###     trainRF_caret_custom_tuner()                     ###
############################################################


trainRF_caret_custom_tuner <- function(y., betas.,
                                       cores = 4L, # some variation of parallel backend needs to be defined
                                       mtry.min = NULL, mtry.max = NULL, length.mtry = 6,
                                       ntrees.min = 1000, ntrees.max = 2000, ntree.by = 500,
                                       nodesize.proc = c(0.01, 0.05, 0.1),
                                       n.cv = 5, n.rep = 1,
                                       p = c(100, 500, 1000, 10000),  # number of values in p should not be more than the number of available cores/threads in the system
                                       p.tuning.brier = T, p.tuning.miscl.err = T, p.tuning.mlogloss = T,
                                       seed, 
                                       allowParallel = T){
  
  # Info block about settings
  set.seed(seed, kind = "L'Ecuyer-CMRG") 
  message("seed: ", seed)
  message("cores: ", cores)
  
  message("\nRun 1.: Random Forest - Hyper parameter tuning with the following number of parameters : ", 
          "\nlength.mtry = ", length.mtry, 
          "; \nntree = ", paste(seq(ntrees.min, ntrees.max, by = ntree.by), sep = ";", collapse = ", "), 
          "; \nlength.nodesize = ", length(c(1, 5, nodesize.proc)),  # default classification =1, def. regression = 5, Kruppa et al. = 10%
          "; \nTotal GRID size: ", length.mtry*length(seq(ntrees.min, ntrees.max, by = ntree.by))*length(c(1, 5, nodesize.proc)), 
          "; \nUsing custom function in {caret} with repeated CV (n.cv = ", n.cv, "; n.rep = ", n.rep, ") ... ", Sys.time())
  
  # Run 1 - Tune RF hyperparameters using grid search (tuning measure = "Accuracy")
  t0 <- system.time(
    # Uses the customRF() function using grid search of the caret training/tuning framework
    rf.hypar.tuner <- subfunc_rf_caret_tuner_customRF(y.subfunc = y., betas.subfunc = betas., 
                                                      mtry.min = mtry.min, mtry.max = mtry.max, length.mtry = length.mtry, 
                                                      ntrees.min = ntrees.min, ntrees.max = ntrees.max, ntree.by = ntree.by,
                                                      nodesize.proc = nodesize.proc,
                                                      n.cv = n.cv, n.rep = n.rep,
                                                      seed = seed, 
                                                      allowParall = allowParallel,
                                                      print.res = T)
    
    
  )
  # Output hyperparameter results in the console
  message("Hyperparameter Tuning Results:", 
          "\n bestTune mtry: ", rf.hypar.tuner$bestTune$mtry, 
          "\n bestTune ntree: ", rf.hypar.tuner$bestTune$ntree, 
          "\n bestTune nodesize: ", rf.hypar.tuner$bestTune$nodesize)
  
  # Run 2 - Fit RF with best tuning parameters for variable selection
  message("\nVariable selection using random Forest with tuned hyperparameters ... ", Sys.time())
  message("Multi-core (available/assigned nr. of threads): ", cores)
  t1 <- system.time(
    rf.varsel <- rfp(xx = betas., y = y.,
                     mc = cores,
                     mtry = rf.hypar.tuner$bestTune$mtry,
                     ntree = rf.hypar.tuner$bestTune$ntree,
                     nodesize = rf.hypar.tuner$bestTune$nodesize,
                     strata = y.,
                     sampsize = rep(min(table(y.)),length(table(y.))), # downsampling to the minority class # smallest class in betas = 8 cases | 4-6 cases
                     importance = TRUE,                              # permutation based mean deacrease in accuracy 
                     replace = FALSE)                                # without replacement see Strobl et al. BMC Bioinf 2007
  )                                                                  # http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25
  
  
  # Extract permutation based importance measure
  message("Variable importance and subsetting of methylation data (betas) using p = ", paste(p, collapse = "; "), " ... ", Sys.time())
  imp.perm <- importance(rf.varsel, type = 1)
  
  # Variable selection ("p" most important variables are selected)
  p.l <- as.list(p)
  message("Reordering betas using multi-core: ", length(p.l))
  or <- order(imp.perm, decreasing = T)
  # Reorder - using mutli-core
  betasy.l <- mclapply(seq_along(p.l), function(i){
    p.l.i <- p.l[[i]]
    betas.[ , or[1:p.l.i]]
  }, mc.preschedule = T, mc.set.seed = T, mc.cores = length(p.l))
  
  
  message("Prediction using 'p' subset of most important variables ... ", Sys.time())
  message("p: ", paste(p, collapse = "; "))  
  message("Multi-core: ", length(betasy.l))
  
  
  set.seed(seed+1, kind = "L'Ecuyer-CMRG")
  t2 <- system.time(
    rf.pred.l <- mclapply(seq_along(betasy.l), function(i){
      randomForest(x = betasy.l[[i]], y = y.,
                   mtry = rf.hypar.tuner$bestTune$mtry,
                   ntree = rf.hypar.tuner$bestTune$ntree,
                   nodesize = rf.hypar.tuner$bestTune$nodesize,
                   strata = y.,
                   sampsize = rep(min(table(y.)), length(table(y.))), # downsampling to the minority class # smallest class in betas = 8 cases | 4-6 cases
                   proximity = TRUE,
                   oob.prox = TRUE,
                   importance = TRUE, # permutation based mean deacrease in accuracy 
                   keep.inbag = TRUE,
                   do.trace = FALSE,
                   seed = seed)
    }, mc.preschedule = T, mc.set.seed = T, mc.cores = length(betasy.l)) # number of cores/threads == number of p variables
  )
  # str(rf.pred.l) # list of 2 # rf objects
  message("\nRun time of training classifier: ")
  print(t2)
  
  # Refit rf.pred.l models on the training data
  score.pred.l <- mclapply(seq_along(rf.pred.l), function(i){
    predict(object = rf.pred.l[[i]], 
            newdata = betas.[ , rownames(rf.pred.l[[i]]$importance)],
            type = "prob")
  },  mc.preschedule = T, mc.set.seed = T, mc.cores = length(rf.pred.l))
  
  # Performance Evaluators for "p" number of predictors tuning
  # Brier score (BS)
  if(p.tuning.brier){
    brier.p.l <- lapply(seq_along(score.pred.l), function(i){
      brier(scores = score.pred.l[[i]], y = y.)
    })
    message("Brier scores: ", paste(brier.p.l, collapse = "; "))
  }
  # Misclassification Error (ME)
  if(p.tuning.miscl.err){
    err.p.l <- lapply(seq_along(score.pred.l), function(i){
      sum(colnames(score.pred.l[[i]])[apply(score.pred.l[[i]], 1, which.max)] != y.) / length(y.)
    })
    message("Misclassif. error on training set (p most important variable): ", paste(err.p.l, collapse = "; "))
  }
  # Multiclass logloss (mLL)
  if(p.tuning.mlogloss){
    mlogloss.p.l <- lapply(seq_along(score.pred.l), function(i){
      mlogloss(scores = score.pred.l[[i]], y = y.)
    })
    message("Multiclass logloss: ", paste(mlogloss.p.l, collapse = "; "))
  }
  # Output optimal "p" number of features (CpG probes) respective to Brier score, log loss and miscl. error
  message("Optimal number of predictor variables:  p(brier) = ", p[[which.min(brier.p.l)]])
  message("Optimal number of predictor variables:  p(mlogl) = ", p[[which.min(mlogloss.p.l)]])
  message("Optimal number of predictor variables:  p(miscl.err) = ", p[[which.min(err.p.l)]])
  
  # Best RF model object  
  message("\nAll p(brier), p(mlogl) and p(miscl.err) models are going to be written into the return object ... ", Sys.time())
  rf.pred.best.brier <- rf.pred.l[[which.min(brier.p.l)]]
  rf.pred.best.err <- rf.pred.l[[which.min(err.p.l)]]
  if(p[[which.min(brier.p.l)]] == p[[which.min(mlogloss.p.l)]]){rf.pred.best.mlogl <- rf.pred.best.brier} # if brier and mlogloss are the same 
  else{rf.pred.best.mlogl <- rf.pred.l[[which.min(mlogloss.p.l)]]}
  
  # Results
  res <- list(rf.pred.best.brier, rf.pred.best.err, rf.pred.best.mlogl, 
              imp.perm, rf.pred.l,
              score.pred.l, brier.p.l, err.p.l, mlogloss.p.l,
              t0, t1, t2)
  return(res)
}

##########################################################################################################################################################################
# Test run / function call
# Sys.time() # [1] "2019-01-29 10:22:36 CET"
# test.run.train.tRF.caret <- trainRF_caret_custom_tuner(y = y[nfolds[[1]][[1]][[1]]$train], betas = betas100[nfolds[[1]][[1]][[1]]$train, ],
#                                                  cores = cores, p = c(50, 100), seed = seed+1, allowParallel = T)
# Sys.time() #[1] "2019-01-29 10:44:35 CET"
# 

# Sys.time()
# [1] "2019-01-29 10:22:36 CET"
# test.run.train.tRF.caret <- trainRF_caret_custom_tuner(y = y[nfolds[[1]][[1]][[1]]$train], betas = betas100[nfolds[[1]][[1]][[1]]$train, ], 
#                                                          +                                                  cores = cores, p = c(50, 100), seed = seed+1, allowParallel = T)
# seed: 20170506
# cores: 11
# 
# Run 1.: Random Forest - Hyper parameter tuning with the following number of parameters : 
#   mtry = 6; 
# ntree = 1000, 1500, 2000; 
# nodesize = 5; 
# Total GRID size: 90; 
# using custom function in {caret} with repeated CV (n.cv = 5; n.rep = 1) ... 2019-01-29 10:22:36
# 2204 samples
# 100 predictors
# 91 classes: 'ETMR', 'MB, G3', 'MB, G4', 'MB, WNT', 'MB, SHH CHL AD', 'MB, SHH INF', 'ATRT, MYC', 'ATRT, SHH', 'ATRT, TYR', 'CNS NB, FOXR2', 'HGNET, BCOR', 'DMG, K27', 'GBM, G34', 'GBM, MES', 'GBM, RTK I', 'GBM, RTK II', 'GBM, RTK III', 'GBM, MID', 'GBM, MYCN', 'CN', 'DLGNT', 'LIPN', 'LGG, DIG/DIA', 'LGG, DNT', 'LGG, RGNT', 'RETB', 'ENB, A', 'ENB, B', 'PGG, nC', 'LGG, GG', 'SCHW', 'SCHW, MEL', 'CPH, ADM', 'CPH, PAP', 'PITAD, ACTH', 'PITAD, FSH LH', 'PITAD, PRL', 'PITAD, STH SPA', 'PITAD, TSH', 'PITAD, STH DNS A', 'PITAD, STH DNS B', 'PITUI', 'EPN, RELA', 'EPN, PF A', 'EPN, PF B', 'EPN, SPINE', 'EPN, YAP', 'EPN, MPE', 'SUBEPN, PF', 'SUBEPN, SPINE', 'SUBEPN, ST', 'CHGL', 'LGG, SEGA', 'LGG, PA PF', 'LGG, PA MID', 'ANA PA', 'HGNET, MN1', 'IHG', 'LGG, MYB', 'LGG, PA/GG ST', 'PXA', 'PTPR, A', 'PTPR, B', 'PIN T,  PB A', 'PIN T,  PB B', 'PIN T, PPT', 'CHORDM', 'EWS', 'HMB', 'MNG', 'SFT HMPC', 'EFT, CIC', 'MELAN', 'MELCYT', 'PLEX, AD', 'PLEX, PED A ', 'PLEX, PED B', 'A IDH', 'A IDH, HG', 'O IDH', 'LYMPHO', 'PLASMA', 'CONTR, ADENOPIT', 'CONTR, CEBM', 'CONTR, HEMI', 'CONTR, HYPTHAL', 'CONTR, INFLAM', 'CONTR, PINEAL', 'CONTR, PONS', 'CONTR, REACT', 'CONTR, WM' 
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold, repeated 1 times) 
# Summary of sample sizes: 1763, 1760, 1763, 1769, 1761 
# Resampling results across tuning parameters:
#   
#   mtry  ntree  nodesize  Accuracy   Kappa      Mean_F1    Mean_Sensitivity  Mean_Specificity
# 5    1000    1        0.9038409  0.9018522  0.8966189  0.9205170         0.9989212       
# 5    1000    5        0.8966113  0.8944688        NaN  0.9074249         0.9988403       
# 5    1000   22        0.7441795  0.7389208        NaN  0.7465884         0.9971345       
# 5    1500    1        0.9006735  0.8986143  0.8972201  0.9167691         0.9988852       
# 5    1500    5        0.9015848  0.8995375        NaN  0.9077591         0.9988957       
# 5    1500   22        0.7454993  0.7403449        NaN  0.7481415         0.9971520       
# 5    2000    1        0.9070116  0.9050798  0.9026972  0.9204093         0.9989565       
# 5    2000    5        0.9007171  0.8986663        NaN  0.9111538         0.9988868       
# 5    2000   22        0.7505377  0.7454472        NaN  0.7498122         0.9972089       
# 6    1000    1        0.9061191  0.9041706  0.9022685  0.9200290         0.9989463       
# 6    1000    5        0.8957146  0.8935579  0.9119036  0.9082723         0.9988307       
# 6    1000   22        0.7510013  0.7459889        NaN  0.7561343         0.9972147       
# 6    1500    1        0.9097887  0.9079121  0.8911206  0.9216391         0.9989875       
# 6    1500    5        0.8938673  0.8916721        NaN  0.9051774         0.9988101       
# 6    1500   22        0.7513878  0.7463396        NaN  0.7558454         0.9972178       
# 6    2000    1        0.9070238  0.9051083  0.8957636  0.9212552         0.9989577       
# 6    2000    5        0.9002564  0.8981978        NaN  0.9103153         0.9988822       
# 6    2000   22        0.7491606  0.7441383        NaN  0.7481216         0.9971955       
# 7    1000    1        0.9052201  0.9032569  0.8916777  0.9148816         0.9989370       
# 7    1000    5        0.9006684  0.8986229        NaN  0.9103614         0.9988870       
# 7    1000   22        0.7409924  0.7357619        NaN  0.7491251         0.9971024       
# 7    1500    1        0.9084280  0.9065402  0.8910812  0.9210399         0.9989735       
# 7    1500    5        0.8952489  0.8930803  0.9248109  0.9085550         0.9988254       
# 7    1500   22        0.7500596  0.7450075        NaN  0.7552762         0.9972038       
# 7    2000    1        0.9056768  0.9037231  0.8947244  0.9182676         0.9989417       
# 7    2000    5        0.8966332  0.8944929        NaN  0.9061477         0.9988409       
# 7    2000   22        0.7668321  0.7620940        NaN  0.7666764         0.9973921       
# 8    1000    1        0.9060992  0.9041534  0.8982389  0.9188520         0.9989461       
# 8    1000    5        0.8970504  0.8949263        NaN  0.9124164         0.9988455       
# 8    1000   22        0.7545789  0.7496490        NaN  0.7569178         0.9972559       
# 8    1500    1        0.9056840  0.9037325  0.8876531  0.9200177         0.9989418       
# 8    1500    5        0.9020412  0.9000086  0.9164322  0.9117157         0.9989017       
# 8    1500   22        0.7496990  0.7447759        NaN  0.7608049         0.9972028       
# 8    2000    1        0.9065901  0.9046603  0.8876885  0.9218502         0.9989521       
# 8    2000    5        0.8993420  0.8972620  0.9148503  0.9112107         0.9988715       
# 8    2000   22        0.7550772  0.7501928        NaN  0.7551037         0.9972625       
# 9    1000    1        0.9052210  0.9032636  0.8792065  0.9153893         0.9989378       
# 9    1000    5        0.8975161  0.8953958  0.9124579  0.9114140         0.9988514       
# 9    1000   22        0.7591582  0.7543300        NaN  0.7573296         0.9973085       
# 9    1500    1        0.9043161  0.9023315  0.8940375  0.9142792         0.9989266       
# 9    1500    5        0.8966187  0.8944828  0.9146749  0.9065586         0.9988412       
# 9    1500   22        0.7514165  0.7464293        NaN  0.7545640         0.9972207       
# 9    2000    1        0.9038762  0.9018884  0.8948283  0.9162424         0.9989221       
# 9    2000    5        0.9002668  0.8982059  0.9193418  0.9145257         0.9988816       
# 9    2000   22        0.7514096  0.7464563        NaN  0.7546177         0.9972213       
# 10    1000    1        0.9011323  0.8990890  0.8887338  0.9134503         0.9988913       
# 10    1000    5        0.8966372  0.8945035        NaN  0.9081350         0.9988412       
# 10    1000   22        0.7509467  0.7460247        NaN  0.7590949         0.9972184       
# 10    1500    1        0.9065694  0.9046442  0.9124569  0.9174933         0.9989528       
# 10    1500    5        0.8993418  0.8972667        NaN  0.9099039         0.9988719       
# 10    1500   22        0.7609897  0.7561858        NaN  0.7658758         0.9973271       
# 10    2000    1        0.9056529  0.9037042  0.8971039  0.9161356         0.9989426       
# 10    2000    5        0.8988935  0.8968033  0.9226419  0.9094451         0.9988664       
# 10    2000   22        0.7586630  0.7538502        NaN  0.7645128         0.9973029       
# Mean_Pos_Pred_Value  Mean_Neg_Pred_Value  Mean_Precision  Mean_Recall  Mean_Detection_Rate
# 0.9108138            0.9989220            0.9108138       0.9205170    0.009932318        
# 0.9100203            0.9988421            0.9100203       0.9074249    0.009852871        
# NaN            0.9971657                  NaN       0.7465884    0.008177797        
# 0.9081340            0.9988869            0.9081340       0.9167691    0.009897511        
# 0.8970391            0.9988984            0.8970391       0.9077591    0.009907526        
# NaN            0.9971802                  NaN       0.7481415    0.008192300        
# 0.9165924            0.9989582            0.9165924       0.9204093    0.009967160        
# 0.9139348            0.9988879            0.9139348       0.9111538    0.009897991        
# NaN            0.9972359                  NaN       0.7498122    0.008247667        
# 0.9209367            0.9989476            0.9209367       0.9200290    0.009957353        
# 0.9171585            0.9988323            0.9171585       0.9082723    0.009843017        
# NaN            0.9972386                  NaN       0.7561343    0.008252761        
# 0.9101130            0.9989895            0.9101130       0.9216391    0.009997678        
# 0.8933881            0.9988119            0.8933881       0.9051774    0.009822718        
# NaN            0.9972442                  NaN       0.7558454    0.008257009        
# 0.9113318            0.9989578            0.9113318       0.9212552    0.009967295        
# 0.8968202            0.9988832            0.8968202       0.9103153    0.009892927        
# NaN            0.9972196                  NaN       0.7481216    0.008232534        
# 0.9135613            0.9989378            0.9135613       0.9148816    0.009947474        
# 0.8974900            0.9988870            0.8974900       0.9103614    0.009897455        
# NaN            0.9971289                  NaN       0.7491251    0.008142774        
# 0.9110282            0.9989737            0.9110282       0.9210399    0.009982726        
# 0.9138349            0.9988271            0.9138349       0.9085550    0.009837900        
# NaN            0.9972301                  NaN       0.7552762    0.008242413        
# 0.9200141            0.9989430            0.9200141       0.9182676    0.009952492        
# 0.8947553            0.9988421            0.8947553       0.9061477    0.009853112        
# NaN            0.9974162                  NaN       0.7666764    0.008426726        
# 0.9166264            0.9989478            0.9166264       0.9188520    0.009957134        
# 0.9160143            0.9988463            0.9160143       0.9124164    0.009857696        
# NaN            0.9972798                  NaN       0.7569178    0.008292076        
# 0.9092106            0.9989426            0.9092106       0.9200177    0.009952571        
# 0.9261990            0.9989034            0.9261990       0.9117157    0.009912540        
# NaN            0.9972270                  NaN       0.7608049    0.008238450        
# 0.9141724            0.9989526            0.9141724       0.9218502    0.009962528        
# 0.9175571            0.9988731            0.9175571       0.9112107    0.009882880        
# NaN            0.9972856                  NaN       0.7551037    0.008297552        
# 0.9045162            0.9989382            0.9045162       0.9153893    0.009947484        
# 0.9182964            0.9988527            0.9182964       0.9114140    0.009862815        
# NaN            0.9973287                  NaN       0.7573296    0.008342398        
# 0.9106861            0.9989283            0.9106861       0.9142792    0.009937540        
# 0.9147728            0.9988420            0.9147728       0.9065586    0.009852952        
# NaN            0.9972460                  NaN       0.7545640    0.008257324        
# 0.9052048            0.9989223            0.9052048       0.9162424    0.009932705        
# 0.9180344            0.9988828            0.9180344       0.9145257    0.009893042        
# NaN            0.9972435                  NaN       0.7546177    0.008257248        
# 0.9101550            0.9988916            0.9101550       0.9134503    0.009902553        
# 0.9081963            0.9988419            0.9081963       0.9081350    0.009853156        
# NaN            0.9972386                  NaN       0.7590949    0.008252162        
# 0.9209044            0.9989527            0.9209044       0.9174933    0.009962301        
# 0.9079210            0.9988722            0.9079210       0.9099039    0.009882877        
# NaN            0.9973502                  NaN       0.7658758    0.008362525        
# 0.9123086            0.9989429            0.9123086       0.9161356    0.009952230        
# 0.9200236            0.9988672            0.9200236       0.9094451    0.009877951        
# NaN            0.9973248                  NaN       0.7645128    0.008336956        
# Mean_Balanced_Accuracy
# 0.9597191             
# 0.9531326             
# 0.8718615             
# 0.9578271             
# 0.9533274             
# 0.8726467             
# 0.9596829             
# 0.9550203             
# 0.8735106             
# 0.9594877             
# 0.9535515             
# 0.8766745             
# 0.9603133             
# 0.9519938             
# 0.8765316             
# 0.9601065             
# 0.9545988             
# 0.8726586             
# 0.9569093             
# 0.9546242             
# 0.8731138             
# 0.9600067             
# 0.9536902             
# 0.8762400             
# 0.9586047             
# 0.9524943             
# 0.8820343             
# 0.9588990             
# 0.9556309             
# 0.8770869             
# 0.9594798             
# 0.9553087             
# 0.8790038             
# 0.9604012             
# 0.9550411             
# 0.8761831             
# 0.9571635             
# 0.9551327             
# 0.8773191             
# 0.9566029             
# 0.9526999             
# 0.8758924             
# 0.9575822             
# 0.9567036             
# 0.8759195             
# 0.9561708             
# 0.9534881             
# 0.8781567             
# 0.9582230             
# 0.9543879             
# 0.8816014             
# 0.9575391             
# 0.9541557             
# 0.8809078             
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were mtry = 6, ntree = 1500 and nodesize = 1.
# Hyperparameter Tuning Results:
#   bestTune mtry: 6
# bestTune ntree: 1500
# bestTune nodesize: 1
# 
# Variable selection using random Forest with tuned hyperparameters ... 2019-01-29 10:43:40
# Multi-core: 11
# Variable importance and subsetting of methylation data (betas) using p = 50; 100 ... 2019-01-29 10:43:49
# Reordering betas using multi-core: 2
# Prediction using 'p' subset of most important variables ... 2019-01-29 10:43:49
# p: 50; 100
# Multi-core: 2
# 
# Run time of training classifier: 
#   user  system elapsed 
# 32.313   0.353  46.074 
# Brier scores: 0.297829848356524; 0.315337570477919
# Misclassif. error on training set (p most important variable): 0.0136116152450091; 0.0122504537205082
# Multiclass logloss: 0.737358628843959; 0.778809892781232
# Optimal number of predictor variables:  p(brier) = 50
# Optimal number of predictor variables:  p(mlogl) = 50
# Optimal number of predictor variables:  p(miscl.err) = 100
# 
# All p(brier), p(mlogl) and p(miscl.err) models are going to be written into the return object ... 2019-01-29 10:44:35
# Warning message:
#   In nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,  :
# There were missing values in resampled performance measures.
# Sys.time()
# [1] "2019-01-29 10:44:35 CET"