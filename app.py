import csv
import json
import os

import plotly
from flask import Flask, render_template, jsonify, session, request, redirect, url_for, send_file
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd
import Methods.Revenue.predict as rv
import Methods.Revenue.plot as pt
import Methods.Consumption.predict as con
import Methods.Consumption.plot as pt_con
import Methods.Generation.predict as gen
import Methods.Generation.plot as pt_gen
import Methods.Revenue.predictGraph as pg
import test
import pyrebase
from google.cloud import storage
import io

app = Flask(__name__)

firebaseConfig = {
  'apiKey': "AIzaSyCAp1r8HXDk0xIw_H6u2U6vTVeq9ge5mLg",
  'authDomain': "powerprophet-61240.firebaseapp.com",
  'projectId': "powerprophet-61240",
  'storageBucket': "powerprophet-61240.appspot.com",
  'messagingSenderId': "1086798285597",
  'appId': "1:1086798285597:web:f80f707e544fac9ea7aff4",
  'measurementId': "G-2JLVNN41CM",
  'databaseURL': 'https://powerprophet-61240-default-rtdb.firebaseio.com'
}
firebase=pyrebase.initialize_app(firebaseConfig)
auth=firebase.auth()
app.secret_key = 'secret'
storage_client = storage.Client("dsgp-383301")
bucket_name = 'dsgpdata'
@app.route('/')
def home():
    revenueTotal = 'revenueTotal.csv'
    Sales = 'Sales.csv'
    residentialRevenue = 'residentialRevenue.csv'

    # Get a handle to the bucket and file
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(revenueTotal)
    blob1 = bucket.blob(Sales)
    blob2 = bucket.blob(residentialRevenue)

    # Download the file contents as a string
    file_contents = blob.download_as_string()
    file_contents1 = blob1.download_as_string()
    file_contents2 = blob2.download_as_string()

    # Convert the file contents to a pandas DataFrame
    data = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
    data01 = pd.read_csv(io.StringIO(file_contents1.decode('utf-8')))
    data02 = pd.read_csv(io.StringIO(file_contents2.decode('utf-8')))
    # data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\revenueTotal.csv")
    # data01 = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\Sales.csv")
    # data02 = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\residentialRevenue.csv")
    predicted_value = round(int(rv.predict( data ))/1000000,2)
    Predicted_value_percentage = rv.expected_percentage_increase( data )
    Predicted_value_differance = rv.difference( data )
    predicted_sales = round(int(rv.predict_Sales(data01))/1000000)
    Predicted_value_percentage_sales = rv.expected_percentage_increase_sales(data01)
    Predicted_value_differance_sales = rv.difference_sales(data01)
    predicted_RR = round(int(rv.predict_RR(data02)) / 1000000, 2)
    Predicted_value_percentage_RR = rv.expected_percentage_RR(data02)
    Predicted_value_differance_RR = rv.difference_RR(data02)
    Plot =pt.ploytPlot()
    Data =pt.chartJsPlot()
    df = data

    df['date'] = pd.to_datetime(df['date'])

    # Group the data by year and sum the total Revenue for each year
    Yearly_Revenue = df.groupby(df['date'].dt.year)['totalRevenue'].sum()

    # Convert the index (which contains the year values) to a list of strings
    labels = Yearly_Revenue.index.astype(str).tolist()

    # Convert the yearly Revenue data to a list of integers
    data = Yearly_Revenue.values.tolist()
    # Sort the yearly Revenue data in descending order
    sorted_sales = Yearly_Revenue.sort_values(ascending=False)

    # Select the top 5 years based on their total Revenue
    top5 = sorted_sales[:5]

    # Convert the index (which contains the year values) to a list of strings
    labels_top5 = top5.index.astype(str).tolist()

    # Convert the yearly Revenue data for the top 5 years to a list of integers
    data_top5 = top5.values.tolist()

    return render_template('dashboard.html', prediction=predicted_value,prediction_02 =predicted_sales,prediction_03=
    Predicted_value_percentage,prediction_04=Predicted_value_differance,prediction_05=Predicted_value_percentage_sales,
                           prediction_06=Predicted_value_differance_sales,prediction_07=Predicted_value_percentage_RR,
                           prediction_08=Predicted_value_differance_RR,prediction_09=predicted_RR,graphJSON=Plot,labels=labels,data=data,
                           data_top5 =data_top5,labels_top5 =labels_top5)
@app.route('/de_revenue')
def home_de():
    data = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\de.csv")
    data01 = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\DeSales.csv")
    data02 = pd.read_csv("C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\DeResidentialRevenue.csv")
    predicted_value = round(int(rv.predict(data)) / 1000000, 2)
    Predicted_value_percentage = rv.expected_percentage_increase(data)
    Predicted_value_differance = rv.difference(data)
    predicted_sales = round(int(rv.predict_Sales(data01)) / 1000000)
    Predicted_value_percentage_sales = rv.expected_percentage_increase_sales(data01)
    Predicted_value_differance_sales = rv.difference_sales(data01)
    predicted_RR = round(int(rv.predict_RR(data02)) / 1000000, 2)
    Predicted_value_percentage_RR = rv.expected_percentage_RR(data02)
    Predicted_value_differance_RR = rv.difference_RR(data02)
    Plot = pt.ploytPlot()
    Data = pt.chartJsPlot()
    df = pd.read_csv('C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\de.csv')

    df['Date'] = pd.to_datetime(df['Date'])

    # Group the data by year and sum the total Revenue for each year
    yearly_sales = df.groupby(df['Date'].dt.year)['totalRevenue'].sum()

    # Convert the index (which contains the year values) to a list of strings
    labels = yearly_sales.index.astype(str).tolist()

    # Convert the yearly Revenue data to a list of integers
    data = yearly_sales.values.tolist()
    # Sort the yearly Revenue data in descending order
    sorted_sales = yearly_sales.sort_values(ascending=False)

    # Select the top 5 years based on their total Revenue
    top5 = sorted_sales[:5]

    # Convert the index (which contains the year values) to a list of strings
    labels_top5 = top5.index.astype(str).tolist()

    # Convert the yearly Revenue data for the top 5 years to a list of integers
    data_top5 = top5.values.tolist()

    return render_template('dashboard.html', prediction=predicted_value, prediction_02=predicted_sales, prediction_03=
    Predicted_value_percentage, prediction_04=Predicted_value_differance,
                           prediction_05=Predicted_value_percentage_sales,
                           prediction_06=Predicted_value_differance_sales, prediction_07=Predicted_value_percentage_RR,
                           prediction_08=Predicted_value_differance_RR, prediction_09=predicted_RR, graphJSON=Plot,
                           labels=labels, data=data,
                           data_top5=data_top5, labels_top5=labels_top5)
# @app.route('/')
# def bar_with_plotly():
#     # Use render_template to pass graphJSON to html
#     return render_template('plot.html', graphJSON=pt.grawPlot())

@app.route('/gen-prediction')
def gen_prediction():
    storage_client = storage.Client("dsgp-383301")
    bucket_name = 'dsgpdata'
    # Load the data from a CSV file
    Generation = 'Generation_dataset_combined.csv'

    # Get a handle to the bucket and file
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob( Generation)

    # Download the file contents as a string
    file_contents = blob.download_as_string()

    # Convert the file contents to a pandas DataFrame
    data = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))

    # data= pd.read_csv('C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\Generation_dataset_combined.csv')
    predicted_value = round(int(gen.predict(data)))
    Predicted_value_percentage = gen.expected_percentage_increase(data)
    Predicted_value_differance = gen.difference(data)
    predicted_value1 = round(int(gen.predict_total(data)))
    Predicted_value_percentage1 = gen.expected_percentage_increase_total(data)
    Predicted_value_differance1 = gen.difference_total(data)

    Plot = pt.ploytPlot()
    Data = pt.chartJsPlot()
    df = data

    df['date'] = pd.to_datetime(df['date'])

    # Group the data by year and sum the total sales for each year
    yearly_sales = df.groupby(df['date'].dt.hour)['megawatthours'].sum()

    # Convert the index (which contains the year values) to a list of strings
    labels = yearly_sales.index.astype(str).tolist()

    # Convert the yearly sales data to a list of integers
    data = yearly_sales.values.tolist()
    # Sort the yearly sales data in descending order
    sorted_sales = yearly_sales.sort_values(ascending=False)

    # Select the top 5 years based on their total sales
    top5 = sorted_sales[:5]

    # Convert the index (which contains the year values) to a list of strings
    labels_top5 = top5.index.astype(str).tolist()

    # Convert the yearly sales data for the top 5 years to a list of integers
    data_top5 = top5.values.tolist()

    return render_template('dashboard_Gen.html',prediction=predicted_value,prediction_03=
    Predicted_value_percentage,prediction_04=Predicted_value_differance,graphJSON=Plot,labels=labels,data=data,
                           data_top5 =data_top5,labels_top5 =labels_top5,prediction_02=predicted_value1,
                           prediction_05=Predicted_value_percentage1,prediction_06=Predicted_value_differance1)
@app.route('/con-prediction')
def con_prediction():
    storage_client = storage.Client("dsgp-383301")
    bucket_name = 'dsgpdata'
    # Load the data from a CSV file
    Consumption = 'Consumption.csv'
    percapita = 'percapita.csv'

    # Get a handle to the bucket and file
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(Consumption)
    blob1 = bucket.blob(percapita)

    # Download the file contents as a string
    file_contents = blob.download_as_string()
    file_contents1 = blob1.download_as_string()

    # Convert the file contents to a pandas DataFrame
    data = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
    data02 = pd.read_csv(io.StringIO(file_contents1.decode('utf-8')))
    predicted_value = round(int(con.predict(data)))
    Predicted_value_percentage = con.expected_percentage_increase(data)
    Predicted_value_differance = con.difference(data)
    predicted_capita = round(int(con.predict_capita(data02)))
    Predicted_value_percentage_capita = con.expected_percentage_capita(data02)
    Predicted_value_differance_capita = con.difference_capita(data02)
    Plot = pt_con.ploytPlot()
    Data = pt_con.chartJsPlot()
    df = data

    df['Date'] = pd.to_datetime(df['Date'])

    # Group the data by year and sum the total sales for each year
    yearly_sales = df.groupby(df['Date'].dt.year)['consumption'].sum()

    # Convert the index (which contains the year values) to a list of strings
    labels = yearly_sales.index.astype(str).tolist()

    # Convert the yearly sales data to a list of integers
    data = yearly_sales.values.tolist()
    # Sort the yearly sales data in descending order
    sorted_sales = yearly_sales.sort_values(ascending=False)

    # Select the top 5 years based on their total sales
    top5 = sorted_sales[:5]

    # Convert the index (which contains the year values) to a list of strings
    labels_top5 = top5.index.astype(str).tolist()

    # Convert the yearly sales data for the top 5 years to a list of integers
    data_top5 = top5.values.tolist()
    # Calculate the yearly consumption
    yearly_sales02 = df.groupby(df['Date'].dt.year)['consumption'].sum()

    # Calculate the difference in consumption between the most recent year and the previous year
    last_year_diff = yearly_sales02.diff().iloc[-1]

    return render_template('dashboard_Con.html', prediction=predicted_value, prediction_02=predicted_capita,
                           prediction_03=
                           Predicted_value_percentage, prediction_04=Predicted_value_differance,
                           prediction_05=Predicted_value_percentage_capita,
                           prediction_06=Predicted_value_differance_capita, prediction_07=last_year_diff, graphJSON=Plot,
                           labels=labels, data=data,
                           data_top5=data_top5, labels_top5=labels_top5)



@app.route('/dataInput',methods=['POST','GET'])
def dataInput():
    if('user' in session):
        return render_template('data.html')
    else:
        return render_template('account.html')

@app.route('/account')
def account():
    return render_template('account.html')

@app.route('/settings')
def settings():
    return render_template('success.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/signup')
def signup():
    return render_template('create_account.html')
@app.route('/Login02')
def login02():
    return render_template('account.html')
@app.route('/login',methods=['POST','GET'])
def login():
    if('user' in session):
        return render_template('my_account.html')
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            user = auth.sign_in_with_email_and_password(email,password)
            session['user'] = email
        except:
            return email
    return render_template('data.html')

@app.route('/create',methods=['POST','GET'])
def create():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            user = auth.create_user_with_email_and_password(email,password)
            session['user'] = email
        except:
            return email
    return render_template('my_account.html')
@app.route('/logout')
def logout():
    if ('user' in session):
        session.pop('user', None)
        return redirect(url_for('home'))
    else:
        return redirect(url_for('home'))

@app.route('/update_consumption')
def update_consumption():
    return render_template('consumption_data.html')
@app.route('/update_revenueTotal')
def update_revenueTotal():
    return render_template('revenueTotal_data.html')

@app.route('/update_residentialRevenue')
def update_residentialRevenue():
    return render_template('residentialRevenue_data.html')

@app.route('/update_sales')
def update_sales():
    return render_template('sales_data.html')

@app.route('/update_percapita')
def update_percapita():
    return render_template('percapita_data.html')

@app.route('/update_generation_dataset_combined')
def update_generation_dataset_combined():
    return render_template('generation_data.html')
def download_byte_range(
    bucket_name, source_blob_name, start_byte, end_byte, destination_file_name
):
    destination_file_path = os.path.join(os.path.expanduser("~"), "Downloads", destination_file_name)
    storage_client = storage.Client("dsgpdata")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_path, start=start_byte, end=end_byte)

@app.route('/download_consumption')
def download_consumption():
    download_byte_range("dsgpdata", "Consumption.csv", 0, None, "Consumption.csv")
    return render_template('data.html')
@app.route('/download_revenueTotal')
def download_revenueTotal():
    download_byte_range("dsgpdata", "revenueTotal.csv", 0, None, "revenueTotal.csv")
    return render_template('data.html')

@app.route('/download_residentialRevenue')
def download_residentialRevenue():
    download_byte_range("dsgpdata", "residentialRevenue.csv", 0, None, "residentialRevenue.csv")
    return render_template('data.html')

@app.route('/download_sales')
def download_sales():
    download_byte_range("dsgpdata", "Sales.csv", 0, None, "Sales.csv")
    return render_template('data.html')

@app.route('/download_percapita')
def download_percapita():
    download_byte_range("dsgpdata", "percapita.csv", 0, None, "percapita.csv")
    return render_template('data.html')

@app.route('/download_generation_dataset_combined')
def download_generation_dataset_combined():
    download_byte_range("dsgpdata", "Generation_dataset_combined.csv", 0, None, "Generation_dataset_combined.csv")
    return render_template('data.html')
@app.route('/insert_consumption',methods=["POST"])
def insert_consumption():
    # get the data from the form
    date = request.form['email']
    consumption = request.form['password']

    # download the CSV file from the bucket
    bucket = storage_client.get_bucket('dsgpdata')
    blob = bucket.blob('Consumption.csv')
    contents = blob.download_as_string()

    # convert the CSV contents to a list of rows
    rows = list(csv.reader(contents.decode().splitlines()))

    # add the new row to the list
    new_row = [date, consumption]
    rows.append(new_row)

    # convert the list of rows back to CSV format
    updated_csv = '\n'.join([','.join(row) for row in rows])

    # upload the updated CSV file back to the bucket
    blob.upload_from_string(updated_csv)

    return render_template('success.html')

@app.route('/insert_revenueTotal',methods=["POST"])
def insert_revenueTotal():
    # get the data from the form
    date = request.form['email']
    revenueTotal = request.form['password']

    # download the CSV file from the bucket
    bucket = storage_client.get_bucket('dsgpdata')
    blob = bucket.blob('revenueTotal.csv')
    contents = blob.download_as_string()

    # convert the CSV contents to a list of rows
    rows = list(csv.reader(contents.decode().splitlines()))

    # add the new row to the list
    new_row = [date, revenueTotal]
    rows.append(new_row)

    # convert the list of rows back to CSV format
    updated_csv = '\n'.join([','.join(row) for row in rows])

    # upload the updated CSV file back to the bucket
    blob.upload_from_string(updated_csv)

    return render_template('success.html')
@app.route('/insert_residentialRevenue',methods=["POST"])
def insert_residentialRevenue():
    # get the data from the form
    date = request.form['email']
    residentialRevenue = request.form['password']

    # download the CSV file from the bucket
    bucket = storage_client.get_bucket('dsgpdata')
    blob = bucket.blob('residentialRevenue.csv')
    contents = blob.download_as_string()

    # convert the CSV contents to a list of rows
    rows = list(csv.reader(contents.decode().splitlines()))

    # add the new row to the list
    new_row = [date, residentialRevenue]
    rows.append(new_row)

    # convert the list of rows back to CSV format
    updated_csv = '\n'.join([','.join(row) for row in rows])

    # upload the updated CSV file back to the bucket
    blob.upload_from_string(updated_csv)

    return render_template('success.html')

@app.route('/insert_sales',methods=["POST"])
def insert_sales():
    # get the data from the form
    date = request.form['email']
    sales = request.form['password']

    # download the CSV file from the bucket
    bucket = storage_client.get_bucket('dsgpdata')
    blob = bucket.blob('Sales.csv')
    contents = blob.download_as_string()

    # convert the CSV contents to a list of rows
    rows = list(csv.reader(contents.decode().splitlines()))

    # add the new row to the list
    new_row = [date, sales]
    rows.append(new_row)

    # convert the list of rows back to CSV format
    updated_csv = '\n'.join([','.join(row) for row in rows])

    # upload the updated CSV file back to the bucket
    blob.upload_from_string(updated_csv)

    return render_template('success.html')

@app.route('/insert_percapita',methods=["POST"])
def insert_percapita():
    # get the data from the form
    date = request.form['email']
    percapita = request.form['password']

    # download the CSV file from the bucket
    bucket = storage_client.get_bucket('dsgpdata')
    blob = bucket.blob('percapita.csv')
    contents = blob.download_as_string()

    # convert the CSV contents to a list of rows
    rows = list(csv.reader(contents.decode().splitlines()))

    # add the new row to the list
    new_row = [date, percapita]
    rows.append(new_row)

    # convert the list of rows back to CSV format
    updated_csv = '\n'.join([','.join(row) for row in rows])

    # upload the updated CSV file back to the bucket
    blob.upload_from_string(updated_csv)

    return render_template('success.html')

@app.route('/insert_generation',methods=["POST"])
def insert_generation():
    # get the data from the form
    date = request.form['email']
    generation = request.form['password']
    total = request.form['password1']
    holiday = request.form['password2']

    # download the CSV file from the bucket
    bucket = storage_client.get_bucket('dsgpdata')
    blob = bucket.blob('Generation_dataset_combined.csv')
    contents = blob.download_as_string()

    # convert the CSV contents to a list of rows
    rows = list(csv.reader(contents.decode().splitlines()))

    # add the new row to the list
    new_row = [holiday,date, total,generation]
    rows.append(new_row)

    # convert the list of rows back to CSV format
    updated_csv = '\n'.join([','.join(row) for row in rows])

    # upload the updated CSV file back to the bucket
    blob.upload_from_string(updated_csv)

    return render_template('success.html')



if __name__ == '__main__':
    app.run(debug=True)