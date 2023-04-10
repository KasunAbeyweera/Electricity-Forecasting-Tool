import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
from google.cloud import storage
import io
import Datasets
import Models
# storage_client = storage.Client("dsgp-383301")
# bucket_name = 'dsgpdata'
# # Load the data from a CSV file
# Consumption = 'Consumption.csv'
# percapita = 'percapita.csv'
#
# # Get a handle to the bucket and file
# bucket = storage_client.bucket(bucket_name)
# blob = bucket.blob(Consumption)
# blob1 = bucket.blob(percapita)
#
#
# # Download the file contents as a string
# file_contents = blob.download_as_string()
# file_contents1 = blob1.download_as_string()
#
# # Convert the file contents to a pandas DataFrame
# data = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
# data02 = pd.read_csv(io.StringIO(file_contents1.decode('utf-8')))
# data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\Consumption.csv")
# data02 = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\percapita.csv")
global_consumption = None
def predict(data):
    # Load the model's architecture from the JSON file
    with open('C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Consumption_Models\\consumption_model_de.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model's weights from the HDF5 file
    loaded_model.load_weights("C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Consumption_Models\\consumption_model_weights_de.h5")

    # Compile the loaded model before using it
    loaded_model.compile(optimizer='adam', loss='mse')

    # Load the data from a CSV file
    # data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\revenueTotal.csv")

    # Extract the last 12 months of data from the DataFrame
    last_12_months = data.tail(13)["consumption"].tolist()

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
    lastMonth = int(data.tail(1)["consumption"].values[0])
    change = round(int(thisMonth - lastMonth))

    if change > 0:
        return "+" + str(change)
    else:
        return str(change)


def expected_percentage_increase(data):
    thisMonth = predict(data)
    lastMonth = int(data.tail(1)["consumption"].values[0])
    increase = thisMonth - lastMonth
    expected_percentage = (increase / lastMonth) * 100

    if expected_percentage > 0:
        return "+" + str(round(expected_percentage, 1)) + "%"
    else:
        return str(round(expected_percentage, 1)) + "%"

def predict_capita(data02):
    # Load the model's architecture from the JSON file
    with open('C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Consumption_Models\\consumption_percapita_Model_de.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model's weights from the HDF5 file
    loaded_model.load_weights("C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Consumption_Models\\consumption_percapita_model_weights_de.h5")

    # Compile the loaded model before using it
    loaded_model.compile(optimizer='adam', loss='mse')


    # Extract the last 12 months of data from the DataFrame
    last_12_months = data02.tail(13)["consumption"].tolist()

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

def difference_capita(data02):
    thisMonth = predict_capita(data02)
    lastMonth = int(data02.tail(1)["consumption"].values[0])
    change = round(int(thisMonth - lastMonth))

    if change > 0:
        return "+" + str(change)
    else:
        return str(change)


def expected_percentage_capita(data02):
    thisMonth = predict_capita(data02)
    lastMonth = int(data02.tail(1)["consumption"].values[0])
    increase = thisMonth - lastMonth
    expected_percentage = (increase / lastMonth) * 100

    if expected_percentage > 0:
        return "+" + str(round(expected_percentage, 1)) + "%"
    else:
        return str(round(expected_percentage, 1)) + "%"


