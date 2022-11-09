from handle import boxcox_df,max_norm
from show import plot_model
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import random

def nsums(l : list):
    sums = {}
    for i,elem in enumerate(l):
        for i2,elem2 in enumerate(l[i:]):
            if elem + elem2 not in sums:
                sums[elem + elem2] = True
    return len(sums)

if __name__ == "__main__":
    df = pd.read_csv("./viinidata/winequality-red.csv",sep=";")
    df = max_norm(df)   # Norm by column max
    #df["quality"] = df["quality"].apply(lambda x : pd.NA if ((int(x) in [5,6]) and (random.random() > 0.1)) else x)
    #df.dropna(inplace=True,axis=0)
    #plt.hist(df["quality"])
    #plt.show()
    # Create correlation matrix
    corr_mat = df.corr()
    #plt.matshow(corr_mat)
    corr_mat.to_excel("./tulokset/corr_mat.xlsx")
    print("New correlation matrix: \n",corr_mat)
    # Modify 'free sulfur dioxide' and 'total sulfur dioxide' so that their distribution is normal
    # This raises each value to the power of the corresponding 'trans' value, and removes rows with a zero value
    #df,trans = boxcox_df(df,cols=["quality"])
    #df,trans = boxcox_df(df)
    #df["free sulfur dioxide"]  = np.log(df["free sulfur dioxide"])
    df = df.astype(float)
    endog = df.pop("quality")
    exog = df
    # Values to remove
    pops = ["residual sugar","citric acid", "fixed acidity", "density","total sulfur dioxide"]
    [exog.pop(k) for k in pops]
    print("EXOG: \n",exog)
    new_corr_mat = exog.corr()
    new_corr_mat.to_excel("./tulokset/tmp-corr_mat.xlsx")
    print("New correlation matrix: \n",new_corr_mat)
    exog = sm.add_constant(exog)
    exog = exog.astype(float)
    lin_model = sm.WLS(endog,exog,hasconst=True,).fit(cov_type="HAC",cov_kwds={"use_correction" : True, "maxlags":1})
    print(lin_model.summary())
    errors = lin_model.predict(exog) - endog
    #errors = errors.apply(lambda x : x**2)
    plt.scatter(endog,errors)
    plt.show()
    
    
    
    
    