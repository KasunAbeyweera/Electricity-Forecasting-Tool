import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import Datasets
import Models

# Load the data from a CSV file

# data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\Generation_dataset_combined.csv")
global_consumption = None
def predict(data):
    # Load the model's architecture from the JSON file
    with open('C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Generation_Models\\gen2_model_de.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model's weights from the HDF5 file
    loaded_model.load_weights("C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Generation_Models\\gen2_model_weights_de.h5")

    # Compile the loaded model before using it
    loaded_model.compile(optimizer='adam', loss='mse')

    # Load the data from a CSV file
    # data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\revenueTotal.csv")

    # Extract the last 12 months of data from the DataFrame
    Last_12_Hours = data.tail(13)["megawatthours"].tolist()

    # Reshape the data into a 2D array with one column
    data_array = np.array(Last_12_Hours).reshape(-1, 1)

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
    ThisHour = predict(data)
    LastHour = int(data.tail(1)["megawatthours"].values[0])
    change = round(int(ThisHour - LastHour))

    if change > 0:
        return "+" + str(change)
    else:
        return str(change)


def expected_percentage_increase(data):
    ThisHour = predict(data)
    lastHour = int(data.tail(1)["megawatthours"].values[0])
    increase = ThisHour - lastHour
    expected_percentage = (increase / lastHour) * 100

    if expected_percentage > 0:
        return "+" + str(round(expected_percentage, 1)) + "%"
    else:
        return str(round(expected_percentage, 1)) + "%"
def predict_total(data):
    # Load the model's architecture from the JSON file
    with open('C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Generation_Models\\gen2_model_de.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model's weights from the HDF5 file
    loaded_model.load_weights("C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Generation_Models\\gen2_model_weights_de.h5")

    # Compile the loaded model before using it
    loaded_model.compile(optimizer='adam', loss='mse')

    # Load the data from a CSV file
    # data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\revenueTotal.csv")

    # Extract the last 12 months of data from the DataFrame
    Last_12_Hours = data.tail(13)["Total"].tolist()

    # Reshape the data into a 2D array with one column
    data_array = np.array(Last_12_Hours).reshape(-1, 1)

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


def difference_total(data):
    ThisHour = predict(data)
    LastHour = int(data.tail(1)["Total"].values[0])
    change = round(int(ThisHour - LastHour))

    if change > 0:
        return "+" + str(change)
    else:
        return str(change)


def expected_percentage_increase_total(data):
    ThisHour = predict(data)
    lastHour = int(data.tail(1)["Total"].values[0])
    increase = ThisHour - lastHour
    expected_percentage = (increase / lastHour) * 100

    if expected_percentage > 0:
        return "+" + str(round(expected_percentage, 1)) + "%"
    else:
        return str(round(expected_percentage, 1)) + "%"



