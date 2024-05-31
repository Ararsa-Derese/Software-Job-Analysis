from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('salary_predictor_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = data['Age']
    gender = data['Gender']
    education_level = data['Education Level']
    job_title = data['Job Title']
    years_of_experience = data['Years of Experience']
    country = data['Country']
    race = data['Race']
    senior = data['Senior']
    
    input_data = pd.DataFrame([[age, gender, education_level, job_title, years_of_experience, country, race, senior]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Country', 'Race', 'Senior'])
    
    prediction = model.predict(input_data)[0]
    # print(prediction
    return jsonify({'predicted_salary': prediction})

if __name__ == '__main__':
    app.run(debug=True)
