 # Importing essential libraries
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn


# Load the LogisticRegression model
model = pickle.load(open('infra_reg.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        cement = float(request.form['a'])
        blast_furnace_slag = float(request.form['b'])
        fly_ash = float(request.form['c'])
        water = float(request.form['d'])
        superplasticizer = float(request.form['e'])
        coarse_aggregate = float(request.form['f'])
        fine_aggregate  = float(request.form['g'])
        age= int(request.form['h'])
        
        
                    
        
        data = np.array([[cement,blast_furnace_slag,fly_ash,water,superplasticizer,coarse_aggregate,fine_aggregate ,age]])
        
        #data.tofile('sample3.csv',sep=',')
        
        my_prediction = model.predict(data)
        
        a = np.array(my_prediction)
        lis = my_prediction.tolist()
        my_prediction = round(lis[0],2)
        #a.tofile('sample1.csv',sep=',')
        return render_template('result.html', prediction=my_prediction)
        

if __name__ == '__main__':
     app.run(debug=True)
