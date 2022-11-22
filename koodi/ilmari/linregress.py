from collections import Counter
from handle import boxcox_df,max_norm
from show import plot_model
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import random
import scipy

def multip_rows(df,ntimes=3,mask_cond=None):
    if mask_cond is None:
        mask_cond = lambda df : (np.abs(scipy.stats.zscore(df)) >= 3).any(axis=1)
    weirds = df[mask_cond(df)]
    print(f"Found {len(weirds)} weird rows.")
    weirds = pd.concat([weirds.copy() for _ in range(ntimes)])
    df = pd.concat([df,weirds])
    return df

def test_model(model,x_test,y_test,norm=True):
    preds = model.predict(x_test)
    preds = round(8*preds)
    y_test = round(8*y_test)
    print(preds)
    print(y_test)
    print("Neural network pedictions",Counter(preds).items())
    print("Actual values",Counter(y_test).items())
    count = 0
    for obs, pred in zip(y_test,preds):
        if obs == pred:
            count += 1
    print("Accuracy of model: ", round(count/len(preds),3))
    errs = preds - y_test
    fig,ax = plt.subplots()
    #print(errs)
    #ax.scatter(y_test,errs)
    ax.hist(errs)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("./viinidata/winequality-white.csv",sep=";")
    df.dropna(inplace=True,axis=0)
    #df = multip_rows(df,ntimes=3,mask_cond=lambda df : df["quality"].isin([1,2,3,7,8,9]))
    #df = multip_rows(df,ntimes=3)
    # Normalize e
    df = max_norm(df)   # Norm by column max
    df = df.astype(float)
    #plt.hist(df["quality"])
    #plt.show()
    # Create correlation matrix
    corr_mat = df.corr()
    #sns.heatmap(corr_mat,
    #            xticklabels=corr_mat.columns.values,
    #            yticklabels=corr_mat.columns.values,
    #            annot=True
    #            )
    plt.show()
    #pp = sns.pairplot(df,hue="quality")
    #plt.matshow(corr_mat)
    corr_mat.to_excel("./tulokset/corr_mat.xlsx")
    print("New correlation matrix: \n",corr_mat)
    # Modify 'free sulfur dioxide' and 'total sulfur dioxide' so that their distribution is normal
    # This raises each value to the power of the corresponding 'trans' value, and removes rows with a zero value
    #df,trans = boxcox_df(df,cols=["quality"])
    #df,trans = boxcox_df(df)
    #df["free sulfur dioxide"]  = np.log(df["free sulfur dioxide"])
    #df = df[(np.abs(scipy.stats.zscore(df)) < 3).all(axis=1)]
    endog = df.pop("quality")
    exog = df
    # Values to remove
    pops = ["residual sugar","citric acid", "fixed acidity", "density","free sulfur dioxide"]
    [exog.pop(k) for k in pops]
    print("EXOG: \n",exog)
    new_corr_mat = exog.corr()
    new_corr_mat.to_excel("./tulokset/tmp-corr_mat.xlsx")
    print("New correlation matrix: \n",new_corr_mat)
    exog = sm.add_constant(exog)
    exog = exog.astype(float)
    
    lin_model = sm.WLS(endog,exog,hasconst=True,).fit(cov_type="HAC",cov_kwds={"use_correction" : True, "maxlags":1})
    print(lin_model.summary())
    test_model(lin_model,exog,endog)
    exit()
    errors = round(10*(lin_model.predict(exog) - endog))
    plt.hist(errors)
    plt.show()
    print(errors)
    #errors = errors.apply(lambda x : x**2)
    plt.scatter(10*endog,errors)
    plt.show()
    