from flask import Flask, request,render_template, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

#--user authentication--

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')



    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html',user=user)
    
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')


#prediction

# Load your dataset (replace 'your_dataset.csv' with the actual file name)
df = pd.read_csv('resources/carsData.csv')
# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['companyName','carModel','engineType', 'showroom'])

# Assuming 'price' is still the target variable
X_encoded = df_encoded.drop('price', axis=1)
y_encoded = df_encoded['price']

# Split the data into training and testing sets
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_encoded, y_train_encoded)

# Train Lasso Regression model
lasso_model = Lasso(alpha=0.1)  # Adjust alpha as needed
lasso_model.fit(X_train_encoded, y_train_encoded)

# Evaluate the models on the test set (optional)

# Predict using Linear Regression model
linear_predictions = linear_model.predict(X_test_encoded)

# Predict using Lasso Regression model
lasso_predictions = lasso_model.predict(X_test_encoded)

# Reshape y_test_encoded to match the predictions
y_test_reshaped = y_test_encoded.values.reshape(-1, 1)

# Calculate the mean squared error
linear_rmse = mean_squared_error(y_test_reshaped, linear_predictions, squared=False)
lasso_rmse = mean_squared_error(y_test_reshaped, lasso_predictions, squared=False)

print(f"Linear Regression RMSE: {linear_rmse}")
print(f"Lasso Regression RMSE: {lasso_rmse}")

# Save the trained models using joblib
from joblib import dump
dump(linear_model, 'linear_model.joblib')
dump(lasso_model, 'lasso_model.joblib')


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

        # Preprocess the data (similar to what you did during model training)
        processed_data = preprocess_data(car_data)

        # Predict using Linear Regression model
        linear_prediction = linear_model.predict([list(processed_data.values())])

        # Predict using Lasso Regression model
        lasso_prediction = lasso_model.predict([list(processed_data.values())])

        return render_template('dashboard.html', linear_prediction=linear_prediction, lasso_prediction=lasso_prediction)

    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)

