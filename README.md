# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along.
A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data. The architecture typically consists of an input layer, one or more hidden layers with activation functions, and an output layer that produces the regression predictions.
This model can capture intricate relationships within data, making it suitable for tasks such as predicting prices, quantities, or any other continuous numerical outputs.

## Neural Network Model
![nn](https://github.com/JaisonRaphael/basic-nn-model/assets/94165957/58fef57c-54c3-4529-878c-3e88bab311a6)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: JAISON RAPHAEL V
### Register Number: 212221230038
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet=gc.open("jaison").sheet1
data=worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'input':'float'})
dataset1 = dataset1.astype({'output':'float'})
dataset1.head()
X = dataset1[['input']].values
y = dataset1[['output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(6,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 120)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information

Include screenshot of the dataset

## OUTPUT

### DATASET INFORMATION

![INFO](https://github.com/JaisonRaphael/basic-nn-model/assets/94165957/9385a987-9d4b-4d77-924f-aaee7b025b2e)


### Training Loss Vs Iteration Plot

![IT](https://github.com/JaisonRaphael/basic-nn-model/assets/94165957/0a1bf3fd-7829-467d-95ab-caf10443d5de)


### Test Data Root Mean Squared Error

![TEST ROOT](https://github.com/JaisonRaphael/basic-nn-model/assets/94165957/917e2fd3-58c3-4e60-89d3-2d7d8c8d2441)


### New Sample Data Prediction

![NEW SAMPLE](https://github.com/JaisonRaphael/basic-nn-model/assets/94165957/c9c0a3db-ee7b-4bcf-b6da-abd6ff417656)


## RESULT

A neural network regression model for the given dataset has been developed Sucessfully.
