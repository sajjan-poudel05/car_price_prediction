from flask import Flask, request, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
from joblib import dump

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

# --user authentication--

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

# Prediction
def preprocess_data(car_data):
    # Load your dataset (replace 'your_dataset.csv' with the actual file name)
    df = pd.read_csv('resources/carsData.csv')
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['companyName', 'carModel', 'engineType', 'showroom'])

    # Assuming 'price' is still the target variable
    X_encoded = df_encoded.drop('price', axis=1)
    y_encoded = df_encoded['price']

    # Split the data into training and testing sets
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42)

    # Train Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train_encoded, y_train_encoded)

    # Train Lasso Regression model
    lasso_model = Lasso(alpha=0.1)  # Adjust alpha as needed
    lasso_model.fit(X_train_encoded, y_train_encoded)

    # Save the trained models using joblib
    dump(linear_model, 'linear_model.joblib')
    dump(lasso_model, 'lasso_model.joblib')

    # Return the models and test data for later use
    return linear_model, lasso_model, X_test_encoded, y_test_encoded

linear_model, lasso_model, X_test_encoded, y_test_encoded = preprocess_data(None)

@app.route('/')
def index():
    return redirect("/login")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', user=user)

    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from the form
        companyName = request.form['companyName']
        carModel = request.form['carModel']
        yearMade = int(request.form['yearMade'])
        kmsRan = int(request.form['kmsRan'])
        engineType = request.form['engineType']
        showroom = request.form['showroom']

        # Create a dictionary of input data
        car_data = {
            'companyName': companyName,
            'carModel': carModel,
            'yearMade': yearMade,
            'kmsRan': kmsRan,
            'engineType': engineType,
            'showroom': showroom
        }

        # Unpack the returned values from preprocess_data
        linear_model, lasso_model, X_test_encoded, y_test_encoded = preprocess_data(car_data)

        # Predict using Linear Regression model
        linear_prediction = linear_model.predict(X_test_encoded)

        # Predict using Lasso Regression model
        lasso_prediction = lasso_model.predict(X_test_encoded)

        # Calculate the mean squared error for each model (optional)
        linear_rmse = mean_squared_error(y_test_encoded, linear_prediction, squared=False)
        lasso_rmse = mean_squared_error(y_test_encoded, lasso_prediction, squared=False)

         # Format predictions as Nepali Rupees
        formatted_linear_prediction = "Rs: {:.0f}".format(linear_prediction[0] * 100000)
        formatted_lasso_prediction = "Rs: {:.0f}".format(lasso_prediction[0] * 100000)

        return render_template(
            'dashboard.html',
            linear_prediction=formatted_linear_prediction,
            lasso_prediction=formatted_lasso_prediction
        )
    return render_template('dashboard.html')


if __name__ == '__main__':
    app.run(debug=True)
