# RIFS
Codes and datasets to reproduce the paper "RIFS: fusing two straws into a diamond"

Abstract：通过随机改变IFS的起始特征来选择那些会被基于统计的算法排在后面的特征。

Hypothesis：两个排名低的特征组合起来可能会对分类算法产生很好的效果

创新：RIFS比现有算法分类准确性更高，并且所用的特征数量更少。同时RIFS对一些异常检测模型也有很好的帮助


效能比较

1.和filter算法比较：Trank, FPR, Wrank

   和RIFS使用相同数量的特征

2.和wrappers算法比较：Lasso, RF, Ridge

   由于wrappers算法自动选择了特征，因此不仅比较准确率还比较选择的特征数



准确率指标：查准率，查全率，F值

使用算法：KNN, NBayes,  SVM, DTree, LR

特征子集性能度量mAcc：五种分类算法在这个特征集上的最大精度(10-fold CV)

环境：Python 2.7.13, Python 3.6.0



算法RIFS

D至少为1，默认为4

起始点默认为特征总数的50%
