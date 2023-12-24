# Import all necessary packages
from flask import Flask, render_template, request 
import pandas as pd 
import numpy as np 
import pickle

# Create an instance of Flask application
application = Flask(__name__)
app = application

# Show the homepage for your app
@app.route('/')
def homepage():
    return render_template('index.html')

# Prediction homepage
@app.route('/predict', methods=['POST'])
def predict_species():
    if request.method=='GET':
        return render_template('index.html')
    else:
        # Load the preprocessor
        with open('notebook/preprocessor.pkl', 'rb') as file1:
            pre = pickle.load(file1)
        # Load the model
        with open('notebook/model.pkl', 'rb') as file2:
            model = pickle.load(file2)
        # Get the input from form 
        sep_len = float(request.form.get('sepal_length'))
        sep_wid = float(request.form.get('sepal_width'))
        pet_len = float(request.form.get('petal_length'))
        pet_wid = float(request.form.get('petal_width'))
        # Convert above values in Dataframe
        xnew = pd.DataFrame([sep_len, sep_wid, pet_len, pet_wid]).T
        xnew.columns = pre.get_feature_names_out()
        # Preprocess the data
        xnew_pre = pre.transform(xnew)
        # Predict the results
        prediction = model.predict(xnew_pre)[0]
        # Probability
        p = model.predict_proba(xnew_pre)
        # Max prob
        prob = round(np.max(p),4)
        return render_template('index.html', prediction=prediction, prob=prob)
    
# Run the application
if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)