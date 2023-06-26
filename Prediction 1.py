###############
#Sourcing Data#
###############

#Call yfinace (used to gather stock data) package in abreviated form which can then be used in code.
import yfinance as yf

# Download historical data for desired ticker
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2020-12-31')
data.to_csv('AAPL.csv')

##########################
#Data Import and Cleaning#
##########################

#Call pandas package (used for data transformation and creating data frames).
import pandas as pd

#Load the data as a data frame into working memory.
data = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)

#Show the first few rows of the data.
print(data.head())

#Check for missing values.
print(data.isna().sum())

#####
#EDA#
#####

#EDA using matplotlib visualisation
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(data['Close'])
plt.title('Closing price of AAPL')
plt.xlabel('Date')
plt.ylabel('Closing Price')
#Show the plot in interactive window.
plt.show()

##################
#Feature Creation#
##################

#Create a 1-day lag variable for closing price
data['Close_Lag1'] = data['Close'].shift(1)
data.dropna(inplace=True)  # drop missing values (there aren't any in this case)

#Create a Month variable
data['Month'] = data.index.month

#Create a Weekday variable
data['Weekday'] = data.index.weekday

########################################
#Splitting into Test and Train Datasets#
########################################
from sklearn.model_selection import train_test_split

#Predictors and target
X = data.drop ('Close', axis=1)
y = data['Close']

#Train-test split creation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#########
#xGboost#
#########
import xgboost as xgb

#Convert the data set into an optimized data structure called Dmatrix that xGboost supports
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#Parameter dictionary specifying base learner
param = {
    'objective': 'reg:squarederror', #Specify the learning task and the corresponding learning objective
    'verbosity': 1 #Printing running messages
}

#Train the model
bst = xgb.train(param, dtrain)

#Use the model to make predictions
y_pred_xgb = bst.predict(dtest)

################
#Neural Network#
################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create the model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

#Fit the model
model.fit(X_train_scaled, y_train, epochs=50, verbose=1)

#Use the model to make predictions
y_pred_nn = model.predict(X_test_scaled).flatten()

##################
#Model Evaluation#
##################
from sklearn.metrics import mean_squared_error

#Create RMSE's for the model
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
print('RMSE for XGBoost:', rmse_xgb)

rmse_nn = mean_squared_error(y_test, y_pred_nn, squared=False)
print('RMSE for Neural Network:', rmse_nn)

#Visualise the model evaluation
plt.figure(figsize=(14, 7))
plt.scatter(y_test.index, y_test, label='Actual')
plt.scatter(y_test.index, y_pred_xgb, label='XGBoost')
plt.scatter(y_test.index, y_pred_nn, label='Neural Network')
plt.title('Comparison of XGBoost and Neural Network Predictions')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

###############################
#Saving and Loading the Models#
###############################
# Save the model
bst.save_model('xgb.model')

# Load the model
bst = xgb.Booster()
bst.load_model('xgb.model')


# Save the model
model.save('nn.model')

# Load the model
from tensorflow.keras.models import load_model
model = load_model('nn.model')

# Remember to also save and load your scaler if you're using it to normalize your data
import joblib
joblib.dump(scaler, 'scaler.gz')  # save
scaler = joblib.load('scaler.gz')  # load

