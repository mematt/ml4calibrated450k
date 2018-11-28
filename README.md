# ml4calibrated450k

### Overview 
This is a repository for comprehensive comparisons of machine learning classifiers and calibration algorithms for personalized cancer diagnostics based on DNA methylation (Illumina 450k} microarray data of molecular neuropathology.

Our comaprisons included random forests (RF), elastic net penalized multinomial logistic regression (ELNET), support vector machines (SVM) and boosted trees (XGBOOST).

For calibration we used i) Platt scaling implemented by logistic regression (LR), Firth's penalized LR; and ii) ridge penalized multinomial regression (rpMLR/MR). 

All algorithms were compared on a molecular neuropathology data set of  brain tumors (n=2801 cases belonging to 91 classes) published in:

Capper, D., Jones, D. T. W., Sill, M. and et al. (2018a). 
Dna methylation-based classification of central nervous system tumours. Nature, 555, 469. 
https://www.nature.com/articles/nature26000

The corresponding Github repository using an rpMLR calibrated RF classifier is available at https://github.com/mwsill/mnp_training

The data set is available in the Gene Expression Omnibus GSE109381 at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109381

***

### Repo content

R notebooks/codes with the implementation of each:
1. Classifier:
+ Random Forests (RF) 
  + vanilla RF (using default settings)
  + tuned RF 
     + Brier score 
     + Misclassification error
     + Multiclass log loss 
  + R package: randomForest, caret
+ Elastic net penalized multinomial logistic regression (ELNET) 
  + concurrent tuning of alpha and lambda 
  + R package: glmnet
+ Support vector machines (SVM)
  + Radial Basis Function kernels (RBF)
  + Linear kernels 
  + R packages: e1071, ksvm (caret), liblineaR; GPU (NVIDIA CUDA accelerated) Rgtsvm
+ Gradient boosted decision trees (XGBOOST)
  + comperehensive tuning of multiple tuning parameters
  + R package: xgboost, caret
2. Calibration/Post-processing Algortihm:
+ Platt scaling 
  + Logistic Regression (LR)
      + R package: glm (base R)
  + Firth's penalized LR (FLR) 
      + R package: brglm 
+ Ridge penalized multinomial logistic regression (rpMLR/MR)
      + R package: glmnet
      
3. We also provide scripts for performance evaluation  such as:
+ Misclassification error (ME)
+ Multiclass AUCH as published by Hand and Till (2001) 
      + R package: HandTill2001
+ Brier score (BS)
+ Mutliclass log loss (mLL) 
  + using the Kaggle formulation https://www.kaggle.com/wiki/LogLoss.
  
### Installation guide 


### Hardware requirements 
Our scripts require (possibly highly) multicore computers. 

The given runtimes were generated using either a workstation with specs of 64 GB RAM, Intel i7 6850k CPU (6 cores/12 thread @ 3.6 GHz) or AWS instances (M.2 64 cores or C.2 16 cores).



    
  
  
  
  


