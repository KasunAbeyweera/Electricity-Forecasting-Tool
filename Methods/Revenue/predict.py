import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import Datasets
import Models

# Load the data from a CSV file

# data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\revenueTotal.csv")
# data01 = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\Sales.csv")
# data02 = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\residentialRevenue.csv")
global_prediction = None
global_sales = None
global_RR = None
def predict(data):
    # Load the model's architecture from the JSON file
    with open('C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Revenue_Models\\model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model's weights from the HDF5 file
    loaded_model.load_weights("C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Revenue_Models\\model_weights.h5")

    # Compile the loaded model before using it
    loaded_model.compile(optimizer='adam', loss='mse')

    # Load the data from a CSV file
    # data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\revenueTotal.csv")

    # Extract the last 12 months of data from the DataFrame
    last_12_months = data.tail(13)["totalRevenue"].tolist()

    # Reshape the data into a 2D array with one column
    data_array = np.array(last_12_months).reshape(-1, 1)

    # Create a MinMaxScaler object to scale the data between 0 and 1
    scaler = MinMaxScaler()

    # Fit the scaler object on the data
    scaler.fit(data_array)

    # Transform the data using the fitted scaler
    scaled_data = scaler.transform(data_array)

    # Generate a sequence of input data for the model to make a prediction on
    n_input = 12
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)

    # Get the input sequence for the first batch
    X, y = generator[0]

    # Reshape the data to match the input shape of the model
    X = X.reshape((1, n_input, 1))

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(X)

    # Inverse transform the prediction to get the actual scale
    actual_prediction = scaler.inverse_transform(prediction)

    # Print the predicted value
    # print("Prediction:", actual_prediction[0][0])
    global_prediction = int(actual_prediction[0][0])
    return actual_prediction[0][0]
    # 2020-08-01,40421937


def difference(data):
    thisMonth = predict(data)
    lastMonth = int(data.tail(1)["totalRevenue"].values[0])
    change = round(int(thisMonth - lastMonth) / 1000000, 2)

    if change > 0:
        return "+" + str(change)
    else:
        return str(change)


def expected_percentage_increase(data):
    thisMonth = predict(data)
    lastMonth = int(data.tail(1)["totalRevenue"].values[0])
    increase = thisMonth - lastMonth
    expected_percentage = (increase / lastMonth) * 100

    if expected_percentage > 0:
        return "+" + str(round(expected_percentage, 1)) + "%"
    else:
        return str(round(expected_percentage, 1)) + "%"


def predict_Sales(data01):
    # Load the model's architecture from the JSON file
    with open('C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Revenue_Models\\model_sales.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model's weights from the HDF5 file
    loaded_model.load_weights("C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Revenue_Models\\model_weights_sales.h5")

    # Compile the loaded model before using it
    loaded_model.compile(optimizer='adam', loss='mse')


    # Extract the last 12 months of data from the DataFrame
    last_12_months = data01.tail(13)["totalSales"].tolist()

    # Reshape the data into a 2D array with one column
    data_array = np.array(last_12_months).reshape(-1, 1)

    # Create a MinMaxScaler object to scale the data between 0 and 1
    scaler = MinMaxScaler()

    # Fit the scaler object on the data
    scaler.fit(data_array)

    # Transform the data using the fitted scaler
    scaled_data = scaler.transform(data_array)

    # Generate a sequence of input data for the model to make a prediction on
    n_input = 12
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)

    # Get the input sequence for the first batch
    X, y = generator[0]

    # Reshape the data to match the input shape of the model
    X = X.reshape((1, n_input, 1))

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(X)

    # Inverse transform the prediction to get the actual scale
    actual_prediction = scaler.inverse_transform(prediction)

    # Print the predicted value
    global_sales = int(actual_prediction[0][0])
    return actual_prediction[0][0]
# 2020-08-31,363989881

def difference_sales(data01):
    thisMonth = predict_Sales(data01)
    lastMonth = int(data01.tail(1)["totalSales"].values[0])
    change = round(int(thisMonth - lastMonth) / 1000000, 2)

    if change > 0:
        return "+" + str(change)
    else:
        return str(change)


def expected_percentage_increase_sales(data01):
    thisMonth = predict_Sales(data01)
    lastMonth = int(data01.tail(1)["totalSales"].values[0])
    increase = thisMonth - lastMonth
    expected_percentage = (increase / lastMonth) * 100

    if expected_percentage > 0:
        return "+" + str(round(expected_percentage, 1)) + "%"
    else:
        return str(round(expected_percentage, 1)) + "%"

def predict_RR(data02):
    # Load the model's architecture from the JSON file
    with open('C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Revenue_Models\\model_RR.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model's weights from the HDF5 file
    loaded_model.load_weights("C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Revenue_Models\\model_weights_RR.h5")

    # Compile the loaded model before using it
    loaded_model.compile(optimizer='adam', loss='mse')


    # Extract the last 12 months of data from the DataFrame
    last_12_months = data02.tail(13)["residentialRevenue"].tolist()

    # Reshape the data into a 2D array with one column
    data_array = np.array(last_12_months).reshape(-1, 1)

    # Create a MinMaxScaler object to scale the data between 0 and 1
    scaler = MinMaxScaler()

    # Fit the scaler object on the data
    scaler.fit(data_array)

    # Transform the data using the fitted scaler
    scaled_data = scaler.transform(data_array)

    # Generate a sequence of input data for the model to make a prediction on
    n_input = 12
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)

    # Get the input sequence for the first batch
    X, y = generator[0]

    # Reshape the data to match the input shape of the model
    X = X.reshape((1, n_input, 1))

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(X)

    # Inverse transform the prediction to get the actual scale
    actual_prediction = scaler.inverse_transform(prediction)

    # Print the predicted value
    global_RR = int(actual_prediction[0][0])
    return actual_prediction[0][0]
# 2020-08-31,363989881

def difference_RR(data02):
    thisMonth = predict_RR(data02)
    lastMonth = int(data02.tail(1)["residentialRevenue"].values[0])
    change = round(int(thisMonth - lastMonth) / 1000000, 2)

    if change > 0:
        return "+" + str(change)
    else:
        return str(change)


def expected_percentage_RR(data02):
    thisMonth = predict_RR(data02)
    lastMonth = int(data02.tail(1)["residentialRevenue"].values[0])
    increase = thisMonth - lastMonth
    expected_percentage = (increase / lastMonth) * 100

    if expected_percentage > 0:
        return "+" + str(round(expected_percentage, 1)) + "%"
    else:
        return str(round(expected_percentage, 1)) + "%"
