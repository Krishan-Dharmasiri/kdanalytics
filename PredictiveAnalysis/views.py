from django.shortcuts import render

import matplotlib 
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import json
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

from io import BytesIO
import base64

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Create your views here.
def lynear_regression(request):
    
    # Load the diabetes dataset
    #advertising =pd.read_csv("C:\Krishantha\Projects\PYTHON\DATA\data.csv")
    advertising = pd.read_csv(os.path.join(BASE_DIR,'PredictiveAnalysis\Data\data.csv'))
    advertising.head()
    advertising_X = advertising[["TV","Radio","Newspaper"]]

    # Split the data into training/testing sets
    advertising_X_train = advertising_X[:-20]
    advertising_X_test = advertising_X[-20:]

    # Split the targets into training/testing sets
    advertising_y_train = advertising[["Sales"]][:-20]
    advertising_y_test = advertising[["Sales"]][-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(advertising_X_train, advertising_y_train)

    # Make predictions using the testing set
    advertising_y_pred = regr.predict(advertising_X_test)
    
    # The coefficients
    #return regr.coef_[0][1]
    
    # The mean squared error
    mse = mean_squared_error(advertising_y_test,advertising_y_pred)
    #return mse
    
    # Explained variance score: 1 is perfect prediction
    variance =  r2_score(advertising_y_test, advertising_y_pred)
    #return variance

    #Plot the scatter chart
    plt.scatter(advertising_y_test, advertising_y_pred,  color='blue', linewidth=3)
    
    #conver the chart into encoded string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    buf.close()
    
    #Return the results as  JSON
    rg = {}
    rg['c1'] = regr.coef_[0][0]
    rg['c2'] = regr.coef_[0][1]
    rg['c3'] = regr.coef_[0][2]
    
    data = {}
    data['regression_coefficients'] = rg
    data['mse'] = mse
    data['r2_score'] = variance
    data['chart'] = "data:image/png;base64," + image_base64
    
    #json_data = json.dumps(data)
    #return json_data
    context = {
        'lynear' : data
    }
    return render(request, 'lynear_regression.html', context) 


