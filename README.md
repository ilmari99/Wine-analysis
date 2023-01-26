# Scool project: Predicting wine quality

This was a school project done in a group of three.

The goal was to assess the effect of different chemical properties on the perceived quality of a wine, based on a dataset from the UCI Machine Learning Repository. [Portuguese Vinho Verde Wine Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality).

My part of the project was creating a model that predicts the quality of a wine based on its chemical properties. We also did linear regression analysis and ordinal logistic regression analysis.

The data was highly imbalanced, so I used a SMOGN algorithm to balance the data. I then compared the performance of different regression models, such as linear regression, random forest, and neural networks on both a balanced and an imbalanced dataset.

The models were compared on both accuracy (rounding the predicted quality to the nearest integer) and on an f1-score with a macro average.

The best models, for both accuracy and f1-score were random forests, with a better f1 score achieved with a balanced dataset, and a better accuracy achieved with an imbalanced dataset.

The best models achieved an accuracy of 69.2 %, an f1-score of 0.48 and a Mean Absolute Error of 0.01.

Best predictors for the quality of a wine were the alcohol content and the amount of sulphates.

The report is in Finnish, at `report.pdf`.

