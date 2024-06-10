from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import pickle
import io
from io import BytesIO
import base64
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained model and encoders/scalers
model = joblib.load('linear_regression_model.pkl')
degree_encode = joblib.load('degree_encoder.pkl')
job_title_encoder = joblib.load('job_title_encoder.pkl')
scaler_age = joblib.load('age_scaler.pkl')
scaler_experience = joblib.load('experience_scaler.pkl')
country_encoder = joblib.load('Country_encoder.pkl') 
arima_model_fit = joblib.load('arima_model')

@app.route('/')
def home():
    return render_template('index.html')
# Load and preprocess the trend prediction dataset
trend_data = pd.read_csv('postings2.csv')
trend_data['date_posted'] = pd.to_datetime(trend_data['date_posted'])
trend_data.set_index('date_posted', inplace=True)
monthly_job_postings = trend_data.resample('ME').size()


@app.route('/predict', methods=['POST'])
def predict():
    # year = int(request.form['year'])
    start_year = request.form['start_year']
    end_year = request.form['end_year']   
    
    # Generate predictions for the entire year
    date_range = pd.date_range(start="2024", end=end_year, freq='ME')
    
    # Calculate the number of steps to forecast
    steps = len(date_range)+24
    # Generate forecasts
    forecast = arima_model_fit.forecast(steps=steps)    
    forecast_index = pd.date_range(start=start_year+"-01-01 00:00:00", periods=steps+1, freq='ME')[1:]
    forecast_series = pd.Series(forecast, index=forecast_index)
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_job_postings, label='Historical Data')
    plt.plot(forecast_series, label='Forecast', color='red')
    print(monthly_job_postings.index[-1])
    plt.axvline(x=monthly_job_postings.index[-1], linestyle='--', color='gray')
    plt.title('Job Postings Forecast')
    plt.xlabel('Date')
    plt.ylabel('Number of Job Postings')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a PNG image in memory
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return jsonify({'plot_url': plot_url})

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    age = int(request.form['age'])
    experience_years = int(request.form['experience_years'])
    degree = request.form['degree']
    job_title = request.form['job_title']
    country = request.form['country']
    if country == "Ethiopia":
        country = "USA"
    print(age, experience_years, degree, job_title)
    # Ensure the input data is present in the encoders
    if degree == 'Bachelors':
        degree_encoded = 0
    elif degree == 'Masters':
        degree_encoded = 1
    else:
        degree_encoded = 2
    # degree_encoded = degree_encode.transform([degree])[0]  
    job_title_encoded = job_title_encoder.transform([str(job_title)])[0]
    print(job_title_encoded)
    country_encoded  = country_encoder.transform([country])[0]
    age_scaled = scaler_age.transform(np.array([[age]]))
    experience_scaled = scaler_experience.transform(np.array([[experience_years]]))
    
    # Combine inputs into a single array
    input_data = np.array([[age_scaled[0][0], experience_scaled[0][0], degree_encoded, job_title_encoded, country_encoded]])
    
    # Predict the salary
    predicted_salary = round(model.predict(input_data)[0],4)
    print(predicted_salary)
    
    return jsonify({'predicted_salary': predicted_salary})
if __name__ == "__main__":
    app.run(debug=True)

