King County Housing Prices - Baseline

In this regression project, I wanted to create a function that makes use of scikit-learns pipelines.
Therefore, the focus is not on EDA or preprocessing. 
The function compares a given set of regressors within pipelines, for the sake of simplicity only using standard scaler currently.
The benefits of the pipelines are, that one could easily adjust this by simply adding for example PCA as an additional step.

Within the comparison function, the different pipelines are cross validated using RMSE and R^2 metrics.
The results are printed with the standard deviation of the results and finally, they are stored in a dataframe and a barplot is
created. The barplot contains the sorted values of the R^2 metric.

