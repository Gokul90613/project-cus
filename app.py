# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

# Load the training data to get the exact columns used during model training
df_1 = pd.read_csv("first_telc.csv")

# Load your model
model = pickle.load(open("model.sav", "rb"))

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    '''
    Input features:
    SeniorCitizen, MonthlyCharges, TotalCharges, Partner, Dependents, 
    MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, 
    TechSupport, Contract, PaperlessBilling
    '''
    
    # Get form inputs
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']

    # Create a DataFrame with the input data
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12]]
    
    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'Partner', 
                                         'Dependents', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                                         'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling'])
    
    # Combine new input with the original training data to get consistent dummy columns
    df_2 = pd.concat([df_1, new_df], ignore_index=True) 

    # Apply one-hot encoding (dummy variables) for features with three categories
    new_df_dummies = pd.get_dummies(df_2[['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 
                                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                          'TechSupport', 'Contract', 'PaperlessBilling']], 
                                    drop_first=False)  # Retain all categories
    
    # Remove any duplicate columns
    new_df_dummies = new_df_dummies.loc[:, ~new_df_dummies.columns.duplicated()]

    # Align the new data with the columns used during model training
    new_df_dummies = new_df_dummies.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Make predictions
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]
    
    if single == 1:
        o1 = "This customer is likely to churn!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        o1 = "This customer is likely to continue!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
        
    return render_template('home.html', output1=o1, output2=o2, 
                           query1=request.form['query1'], 
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'], 
                           query6=request.form['query6'], 
                           query7=request.form['query7'], 
                           query8=request.form['query8'], 
                           query9=request.form['query9'], 
                           query10=request.form['query10'], 
                           query11=request.form['query11'], 
                           query12=request.form['query12'])

app.run(debug=True)
