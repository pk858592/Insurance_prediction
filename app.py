from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('gradient_boosting_model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods =['POST'])
def predict_placement():
    Age = int(request.form.get('Age'))
    sex = int(request.form.get('sex'))
    bmi = float(request.form.get('bmi'))
    smoker = int(request.form.get('smoker'))
    region = int(request.form.get('region'))

    # prediction
    result = model.predict(np.array([[Age,sex,bmi,smoker,region]]))



    return render_template('index.html',result = result)
if __name__ == '__main__':
    app.run(debug=True)