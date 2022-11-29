import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import scipy
import numpy as np
from typing import Tuple,List
import seaborn as sb
from handle import standard_transformation

RW_DATA = pd.read_csv("./viinidata/winequality-red.csv",sep=";")
#WW_DATA = pd.read_csv("./viinidata/winequality-white.csv",sep=";")

def create_correlation_heatmap(df,show=True,save = ""):
    sb.heatmap(df.corr())
    

def create_pair_plot(df,show = True, save="", categ_quality = False, pairplot_kwargs={}):
    if categ_quality:
        df["quality2"] = df["quality"].apply(lambda x : (x > 4) + (x >  6))
        print(df.head())
    endog = pd.DataFrame()
    endog["quality"] = df.pop("quality")
    endog["quality2"] = df.pop("quality2")
    exog = df
    exog = standard_transformation(exog)
    df = pd.concat([exog,endog],axis=1)
    print(df.head())
    pair_grid = sb.pairplot(df,hue="quality2",**pairplot_kwargs)
    pair_grid.tick_params(axis="both",which="both",reset = True,labelrotation=60)
    #pair_grid.fig.draw(
    #pair_grid.fig.canvas.get_renderer()) 
    #pair_grid.set(xticklabels=list(df.columns))
    #for ax in pair_grid.axes.flat:
    #    ax.tick_params("x",rotation=60)
    #    ax.tick_params("y",rotation=60)
    if show:
        plt.show()
    
if __name__ == "__main__":
    create_pair_plot(RW_DATA,categ_quality=True,pairplot_kwargs={"diag_kind":"kde",})#"diag_kind":"quality"})