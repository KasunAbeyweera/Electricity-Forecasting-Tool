import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import json
import numpy as np


# Load the model's architecture from the JSON file
with open('C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Consumption_Models\\consumption_model_de.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the model's weights from the HDF5 file
loaded_model.load_weights("C:\\Users\\kasun\\Downloads\\flaskProject\\Models\\Consumption_Models\\consumption_model_weights_de.h5")

# Compile the loaded model before using it
loaded_model.compile(optimizer='adam', loss='mse')

# Load the data from a CSV file
data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\Consumption.csv")
def predictGraph():
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
    print("Prediction:", actual_prediction[0][0])

    # Create a new array with the predicted value and the previous 11 values
    predicted_data = np.append(data["consumption"].tail(12).values, actual_prediction[0][0])

    # Remove the first element (the oldest actual value) from the predicted data array
    predicted_data = predicted_data[1:]

    # Get the date range for the predicted data
    start_date = pd.to_datetime(data["Date"].tail(12).values[0])
    end_date = start_date + pd.DateOffset(months=12)
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")

    # Get the actual past values data (y-axis data)
    actual_data = data["consumption"].tail(12)

    data_json = json.dumps({
        "labels": date_range.format(),
        "actual_data": [float(x) for x in actual_data.tolist()],  # convert to list of floats
        "predicted_data": [float(x) for x in predicted_data.tolist()],  # convert to list of floats
        "prediction_value": float(actual_prediction[0][0])  # convert to float
    })

    return data_json
