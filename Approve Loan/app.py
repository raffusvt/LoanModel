from flask import Flask, request, jsonify
from flask_cors import CORS # import cors
import joblib
import numpy as np
# loan the pre-trained model

model = joblib.load('loan_prediction_model.pkl')

app = Flask(__name__)
CORS(app) #enable cors
@app.route('/predict',method=['POST'])
def predict():
    data = request.get_json()
    income = data['income']
    credit_score= data['credit_score']
    
    #prepare the input data for prediction
    input_data = np.array([[income,credit_score]])

    # make prdiction
    prediction= model.predict(input_data)

    # convert prediction to readable message
    result="Approved" if prediction[0] ==1 else "Not Approved"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)