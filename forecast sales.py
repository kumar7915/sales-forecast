#!/usr/bin/env python
# coding: utf-8

# # Cement Sales Forecasting 

# Import required libraries

# In[61]:


import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.metrics import mean_absolute_percentage_error
import math


# In[62]:


data = pd.read_excel(r"C:\Users\psaik\OneDrive\Desktop\All India_Features.xlsx")


# In[63]:


data


# In[64]:


data.info()


# In[65]:


data.describe()


# In[66]:


data.isna().sum()


# In[67]:


data.duplicated().sum()


# In[68]:


data[['GDP_Realestate_Rs_Crs', 'GDP_Construction_Rs_Crs']] = data[['GDP_Realestate_Rs_Crs', 'GDP_Construction_Rs_Crs']].astype(float)


# In[69]:


data['Date'] =  pd.to_datetime(data['Date'])


# In[70]:


plt.hist(data['Sales_Quantity_Milliontonnes'])
plt.title("Sales_Quantitu_Milliontonnes")


# In[71]:


sns.boxplot(data['GDP_Construction_Rs_Crs'])


# In[72]:


sns.boxplot(data['Oveall_GDP_Growth%'])


# In[73]:


sns.boxplot(data['Coal_Milliontonne'])


# In[74]:


sns.boxplot(data['Home_Interest_Rate'])


# Winsorization to remove outliers

# In[75]:


from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['GDP_Construction_Rs_Crs', 'Oveall_GDP_Growth%',
                    'Coal_Milliontonne', 'Home_Interest_Rate'])

data[['GDP_Construction_Rs_Crs', 'Oveall_GDP_Growth%', 'Coal_Milliontonne', 'Home_Interest_Rate']] = winsor.fit_transform(data[['GDP_Construction_Rs_Crs',
                                                                          'Oveall_GDP_Growth%', 'Coal_Milliontonne', 'Home_Interest_Rate']])
data


# In[76]:


plt.plot(data['Date'], data['Sales_Quantity_Milliontonnes'], label = 'Sales')
plt.title('2015-2022 Cement Sales')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.grid()
plt.legend()


# In[77]:


decomposition = seasonal_decompose(data['Sales_Quantity_Milliontonnes'], model='addititive', period= 12)
decomposition.plot()


# Adfuller test to check stationarity

# In[78]:


for i in range(len(data.columns)):
    result = adfuller(data[data.columns[i]])
    
    if result[1] > 0.05 :
        print('{} - series is not stationary'.format(data.columns[i]))
    else:
        print('{} - series is stationary'.format(data.columns[i]))


# EDA report using AUTO EDA

# In[79]:


from pandas_profiling import ProfileReport

profile = ProfileReport(data, tsmode=True, sortby="Date")
profile


# Splitting the data

# In[80]:


train = data.iloc[:-12]
test = data.iloc[-12:]
train, test


# Rename date and sales columns
# 
# Prophet expect date column and output column names to be ds and y

# In[81]:


train = train.rename(columns={'Sales_Quantity_Milliontonnes': 'y', 'Date':'ds'})
train


# In[82]:


test = test.rename(columns={'Sales_Quantity_Milliontonnes': 'y', 'Date':'ds'})
test


# Initiating model
# 
# Univariate model

# In[83]:


model_uv = Prophet()


# In[84]:


model_uv.fit(train)


# In[85]:


train_uv_forecasts = model_uv.predict(train)
train_uv_forecasts


# In[86]:


mse = mean_squared_error(train_uv_forecasts['yhat'], train['y'])
rmse = math.sqrt(mse)
mae = mean_absolute_error(train_uv_forecasts['yhat'], train['y'])
mape = mean_absolute_percentage_error(train_uv_forecasts['yhat'], train['y'])
mse, rmse, mae, mape


# In[87]:


test_uv_forecasts = model_uv.predict(test)
test_uv_forecasts


# In[88]:


mse = mean_squared_error(test_uv_forecasts['yhat'], test['y'])
rmse = math.sqrt(mse)
mae = mean_absolute_error(test_uv_forecasts['yhat'], test['y'])
mape = mean_absolute_percentage_error(test_uv_forecasts['yhat'], test['y'])
mse, rmse, mae, mape


# Multivariate model

# In[89]:


model = Prophet( growth='linear', yearly_seasonality= True, weekly_seasonality= False,
                daily_seasonality= False, holidays=None, seasonality_mode='additive',)


# Adding regressors to the model built

# In[90]:


model.add_regressor('GDP_Construction_Rs_Crs')
model.add_regressor('GDP_Realestate_Rs_Crs')
model.add_regressor('Oveall_GDP_Growth%')
model.add_regressor('Water_Source')
model.add_regressor('Limestone')
model.add_regressor('Coal_Milliontonne')
model.add_regressor('Home_Interest_Rate')
model.add_regressor('Trasportation_Cost')
model.add_regressor('Order_Quantity_Milliontonnes')
model.add_regressor('Unit_Price')
model.fit(train)


# In[44]:


train_mv_forecasts = model.predict(train)
train_mv_forecasts


# In[91]:


plot_plotly(model, train_mv_forecasts, xlabel = 'Date', ylabel = 'Sales_Milliontonnes')


# In[92]:


model.plot_components(train_mv_forecasts)


# In[93]:


mse = mean_squared_error(train_mv_forecasts['yhat'], train['y'])
rmse = math.sqrt(mse)
mae = mean_absolute_error(train_mv_forecasts['yhat'], train['y'])
mape = mean_absolute_percentage_error(train_mv_forecasts['yhat'], train['y'])
mse, rmse, mae, mape


# Compare Univariate vs Multivariate forecasts

# In[94]:


plt.plot(train['ds'], train['y'], color = 'red', label = 'actual')
plt.plot(train['ds'], train_uv_forecasts['yhat'], color = 'blue', label = 'univariate')
plt.plot(train['ds'], train_mv_forecasts['yhat'], color = 'green', label = 'multiivariate')
plt.title('Actual values vs Univariate and Multivariate Forecasts')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.grid()
plt.legend()


# In[95]:


test_mv_forecasts = model.predict(test)
test_mv_forecasts


# In[96]:


plot_plotly(model, test_mv_forecasts,  xlabel = 'Date', ylabel = 'Sales_Milliontonnes')


# In[97]:


model.plot_components(test_mv_forecasts)


# In[98]:


test_forecasts = pd.DataFrame(test_mv_forecasts[['yhat', 'yhat_upper', 'yhat_lower']])
test_forecasts


# In[99]:


mse = mean_squared_error( test['y'], test_mv_forecasts['yhat'])
rmse = math.sqrt(mse)
mae = mean_absolute_error( test['y'], test_mv_forecasts['yhat'])
mape = mean_absolute_percentage_error( test['y'], test_mv_forecasts['yhat'])
mse, rmse, mae, mape


# In[104]:


data1 = pd.read_excel(r"C:\Users\psaik\OneDrive\Desktop\final data with future.xlsx")
data1


# In[105]:


forecast = model.predict(data1)


# In[106]:


plot_plotly(model, forecast,  xlabel = 'Date', ylabel = 'Sales_Milliontonnes')


# In[107]:


model.plot_components(forecast)


# In[108]:


Forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)


# In[109]:


Forecast_df.rename(columns = {'ds' : 'Date', 'yhat' : 'Sales_Forecast', 'yhat_upper' : 'Sales_Max_Forecast', 'yhat_lower' : 'Sales_Min_Forecast'}, inplace = True)


# In[110]:


Forecast_df


# In[ ]:




