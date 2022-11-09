import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from utils import plot_model
from ModelSet import ModelSet
import matplotlib.pyplot as plt
import scipy
import numpy as np
from typing import Tuple,List

df = pd.read_csv("./viinidata/winequality-red.csv",sep=";")

# Normalize data
df_max_min = [(max(df[col]),min(df[col])) for col in df.columns]
#df = df.apply(lambda x : x / max(x))

# Calculate average of each variable at each unique quality value
#df = df.groupby(by="quality",as_index=False).mean()

new_corr_mat = df.corr()
#new_corr_mat["ind"] = df.columns
#new_corr_mat.set_index("ind",drop=True)
new_corr_mat.to_excel("./tulokset/corr_mat.xlsx")
print("New correlation matrix: \n",new_corr_mat)

print(df.describe())

df,trans = boxcox_df(df,cols=["free sulfur dioxide","total sulfur dioxide"])
endog = df.pop("quality")
exog = df
# Remove variables correlated with other variables
pops = ["residual sugar","fixed acidity","citric acid", "density"]
#pops = ["citric acid"]
[exog.pop(k) for k in pops]
new_corr_mat = exog.corr()
new_corr_mat.to_excel("./tulokset/tmp-corr_mat.xlsx")
print("New correlation matrix: \n",new_corr_mat)
exog = sm.add_constant(exog)
exog = exog.astype(float)
lin_model = sm.WLS(endog,exog,hasconst=True).fit()
print(lin_model.summary())
errors = lin_model.predict(exog) - endog
errors = errors.apply(lambda x : x**2)
plt.scatter(endog,errors)
plt.show()
