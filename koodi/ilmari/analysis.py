import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import scipy
import numpy as np
from typing import Tuple,List
import seaborn as sb
from linregress import white_wine_linmodel, red_wine_linmodel
from handle import standard_transformation,make_smogn,add_noise,max_norm

RW_DATA = pd.read_csv("./viinidata/winequality-red.csv",sep=";")
#WW_DATA = pd.read_csv("./viinidata/winequality-white.csv",sep=";")
RW_POPS = ["residual sugar","citric acid", "fixed acidity", "density","free sulfur dioxide"]
WW_POPS = ["residual sugar","total sulfur dioxide","citric acid","pH","density",]

def create_correlation_heatmap(df,show=True,wine="red"):
    #pops = ["fixed acidity","density"]
    if wine == "red":
        pops = RW_POPS
    if wine == "white":
        pops = WW_POPS
    [df.pop(k) for k in pops]
    ax = sb.heatmap(100*df.corr(),annot=True,fmt=".0f",)
    ax.set_title("White wine correlation plot (%)")
    plt.xticks(rotation = 45)
    if show:
        plt.show()

def create_pair_plot(df,show = True, save="", categ_quality = False, wine = "red", pairplot_kwargs={}):
    if categ_quality:
        df["quality2"] = df["quality"].apply(lambda x : (x > 4) + (x >  6))
        print(df.head())
    if wine == "red":
        pops = RW_POPS#["fixed acidity","citric acid","density"]#["residual sugar","citric acid", "fixed acidity", "density","free sulfur dioxide"]
    elif wine == "white":
        pops = WW_POPS
    [df.pop(k) for k in pops]
    pg = sb.pairplot(df,hue="quality2",palette=sb.color_palette("tab10",3),**pairplot_kwargs)
    #pg = sb.PairGrid(df,hue="quality2")
    #pg.map_upper(sb.scatterplot)
    #pg.map_lower(sb.kdeplot)
    #pg.map_diag(sb.kdeplot)
    #pg.add_legend()
    pg.figure.align_labels()
    #pg.figure.autofmt_xdate(rotation=45,which="both")

    #pair_grid = sb.pairplot(df,hue="quality2",**pairplot_kwargs)
    #pair_grid.tick_params(axis="both",which="both",reset = True,labelrotation=60)
    #pair_grid.fig.draw(
    #pair_grid.fig.canvas.get_renderer()) 
    #pair_grid.set(xticklabels=list(df.columns))
    #for ax in pg.axes.flat:
    #    ax.tick_params("x",rotation=60)
    #    ax.tick_params("y",rotation=60)
    if save:
        pg.figure.savefig(save)
    if show:
        plt.show()
        
def apply_smogn(df):
    y = df.pop("quality")
    y,_ = add_noise(y,noise_division=80)
    df = pd.concat([df,y],axis=1)
    df,y = make_smogn(df,y_header="quality")
    df = pd.concat([df,y],axis=1)
    print(df)
    print(df.describe())
    return df

def fit_linmodel(df,wine="red",pops = "default"):
    if wine == "red" and not isinstance(pops,(list,tuple)):
        pops = RW_POPS
    elif wine == "white" and not isinstance(pops,(list,tuple)):
        pops = WW_POPS
    exog = df
    endog = df.pop("quality")
    model = red_wine_linmodel(exog,endog,pops = pops)
    print(exog.corr())
    return model
    
if __name__ == "__main__":
    #create_correlation_heatmap(WW_DATA)
    #WW_DATA = apply_smogn(WW_DATA)
    #create_correlation_heatmap(WW_DATA,wine="white")
    #create_pair_plot(RW_DATA,wine="red",categ_quality=True,save="red-wine-pairplot.png",pairplot_kwargs={"diag_kind":"kde",})#"diag_kind":"quality"})
    #RW_DATA = max_norm(RW_DATA,inplace=False)
    #print(WW_DATA.describe())
    fit_linmodel(RW_DATA,wine="red",pops="default")
