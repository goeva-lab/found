from .adapters import Pipeline
from .methods import kmeans_bin, log_reg, norm_log1p, run_pca, score_deg, score_ks

LogNormPCALogRegKMeansKSScore = Pipeline(norm_log1p, run_pca, log_reg, kmeans_bin, score_ks, True)
"""
:py:class:`~found.adapters.Pipeline` consisting of the following steps:

- log1p normalize w/ size factor scaling
- run PCA
- run logistic regression
- k-means on the regression probability to binarize
- score based on KS stat between probabilities in new labels
"""


LogNormPCALogRegKMeansDEGScore = Pipeline(norm_log1p, run_pca, log_reg, kmeans_bin, score_deg, True)
"""
:py:class:`~found.adapters.Pipeline` consisting of the following steps:

- log1p normalize w/ size factor scaling
- run PCA
- run logistic regression
- k-means on the regression probability to binarize
- score based on number of DEGs between new refined labels
"""
