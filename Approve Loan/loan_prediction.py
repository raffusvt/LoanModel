import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

#sample data
data = {
    'Name': ['Alice','Bob','Charlie','David','Eva','Frank','Grace','Henry','Isla','jack'] * 5,
    'Income': [30000,45000,50000,60000,35000,48000,52000,58000,47000,39000] * 5,
    'Credit_Score': [650,700,720,680,620,600,750,710,640,730] * 5,
    'Loan_Approved': [1,1,1,1,0,0,1,1,0,1]
}
#Create dataframe
df=pd.DataFrame(data)

# Feature and target variable
X=df[['Income','Credit_score']]
Y=df['Loan_Approved']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model,'loan_prediction_model.pkl')

print("Model trained and saved!")
