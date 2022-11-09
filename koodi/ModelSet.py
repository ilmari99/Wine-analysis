from matplotlib import style
import pandas as pd
import sys, os
import numpy as np
from utils import concat_clean_separate,plot_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

class ModelSet:
    """A class that contains a dictionary of models. The dictionary is keyed by the
    the Dataframes column headers. The models are fitted to the values in the column versus the
    x variable, defined with 'x_header : str'. Defaults to the first column.
    """
    models = {}
    y_header = ""
    df = None
    measures = None
    def __init__(self, df : pd.DataFrame,
                 y_header : str = ""):
        """Returns a ModelSet object. Creates a dictionary of models from the dataframe.
        Uses the first column, or the column identified by argument 'y_header', as the y variable.
        Fits the models to the values in the column versus the y variable.
        """
        df = df.copy()         # Make a shallow copy of the original dataframe
        # If x_header is not given, use the first column
        if y_header == "":
            y_header = df.columns[0]
        self.y_header = y_header
        # Create a shallow copy of the initial dataframe
        self.df = df.copy(deep=False)
        y = df.pop(y_header)                       # Get the dependent variable
        x = df                                      # Get the independent variables
        self.measures = y.to_list()                 # Get the years that are measured, and not predicted
        self.models = self.create_model_dict(x,y.to_frame())   # Create the header-model dictionary
        
    def create_model_dict(self,x,y):
        """ Fit multiple y variables independently to the same x variable. Returns a dictionary with
        y columns headers as keys and the fitted OLS statsmodels model as values.
        """
        # append truth values of conditions to list and which truth values are False
        fails = []
        fails.append(isinstance(y,pd.DataFrame) and isinstance(x,pd.DataFrame))
        fails.append(y.shape[0] == x.shape[0])
        for i,f in enumerate(fails):
            if f:
                continue
            if i == 0:
                print("x",type(x),"y",type(y))
                raise TypeError("x and y must be pandas dataframes or series")
            if i == 1:
                print("x",x.shape,"y",y.shape)
                print(x)
                print(y)
                raise ValueError("y must have the same number of rows as x has elements")
        model_dict = {}
        # Loop through the columns of y and fit a model its columns versus the x variable
        for col in x.columns:
            xx = x.loc[:,col]
            xx,yy = concat_clean_separate(xx,y) # Remove NaN values
            model = fit_statmodel(yy,xx)
            model_dict[col] = model
        return model_dict
    
    def _predict_from_model_set(self,x):
        """ Predict a single y value from the dictionary of models.
        """
        if not isinstance(x, (pd.DataFrame,pd.Series)):
            try:
                x = pd.DataFrame([x])
            except ValueError:
                raise ValueError("x must be a single value, casting unsuccesful")
        assert x.shape[1] == 1, "x must be a single value"
        y_header = self.y_header
        y = pd.DataFrame(columns=[y_header])        # Create a dataframe with a single column
        y[y_header] = x                             # Add the x value to the dataframe
        x = sm.add_constant(x,has_constant="add")   # Add a constant to the predictor variable
        for col,model in self.models.items():       # Loop through the keys and models and predict y
            p = model.predict(x)
            y[col] = p                              # Add the prediction to the dataframe
        # The result is a dataframe with the x value as the first column and the predicted y values (along with headers) as the other columns
        return y
    
    def _predict_conf_int(self,x, alpha=0.05):
        """ Return the confidence of a single prediction, and the prediction itself
        Returns an nX3 dataframe with the prediction, and the confidence interval."""
        out = self._predict_from_model_set(x)
        lb = pd.DataFrame(columns=out.columns)
        ub = pd.DataFrame(columns=out.columns)
        x = sm.add_constant(pd.DataFrame([x]),has_constant="add")
        for col in out.columns:
            if col == self.y_header:
                continue
            print("original",out[col])
            p = self.models[col].get_prediction(x)
            fr = p.summary_frame(alpha=alpha)
            lb[col] = fr.loc[:,"obs_ci_lower"]
            ub[col] = fr.loc[:,"obs_ci_upper"]
        out = pd.concat([out,lb,ub],axis=0)
        return out
            
    def get_row_or_prediction(self,vals,get_ci=False,alpha=0.05):
        """ Returns a dataframe with measured and/or predicted values.
        An empty dataframe is created
        The values in vals are looped through, and if they have a measurement, the row is added to the dataframe.
        Else, the prediction is made and the returned row is added to the dataframe.
        """
        if get_ci and len(vals)>1:
            raise NotImplementedError("get_ci can only be used with a single value")
        out = pd.DataFrame()
        for v in vals:
            # If x is a value with measurement, get a copy of the row and add it to the dataframe
            if v in self.measures:
                row = self.df.loc[self.df[self.y_header]==v].copy(deep=True)
            # Else calculate the prediction
            elif get_ci:
                row = self._predict_conf_int(v,alpha=alpha)
            else:
                row = self._predict_from_model_set(v)
            out = pd.concat([out,row],axis=0)
        return out

def fit_statmodel(y,x,fix_intercept=False):
    model = sm.OLS(y,sm.add_constant(x)).fit()
    return model