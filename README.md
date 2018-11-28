# ml4calibrated450k
This is a repository for comprehensive comparisons of machine learning classifiers and calibration algorithms for personalized cancer diagnostics based on DNA methylation (Illumina 450k} microarray data of molecular neuropathology.

Our comaprisons included random forests (RF), elastic net penalized multinomial logistic regression (ELNET), support vector machines (SVM) and boosted trees (XGBOOST).

For calibration we used i) Platt scaling implemented by logistic regression (LR), Firth's penalized LR; and ii) ridge penalized multinomial regression (rpMLR). 

All algorithms were compared on a molecular neuropathology data set of  brain tumors (n=2801 cases belonging to 91 classes) published in:

Capper, D., Jones, D. T. W., Sill, M. and et al. (2018a). 
Dna methylation-based classification of central nervous system tumours. Nature, 555, 469. 
https://www.nature.com/articles/nature26000

The corresponding Github repository using an rpMLR calibrated RF classifier is available at https://github.com/mwsill/mnp_training

The data set is available in the Gene Expression Omnibus GSE109381 at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109381



