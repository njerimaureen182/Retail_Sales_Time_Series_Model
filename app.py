import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import fredapi as fa
from local_settings import fred as settings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title = 'Sales Forecast App', layout = 'wide')

#load the data for plotting actuals
@st.cache_resource
def load_data():

    #import the API key
    fred = fa.Fred(settings['API_KEY'])

    #fetch and structure the data
    df = fred.get_series('RSXFS').to_frame(name='RSXFS').reset_index()
    df = df.rename(columns={'index':'date'})
    df['date']  = pd.to_datetime(df['date'])

    #set' date' as an index
    df.set_index('date', inplace=True)

    return df

#refit SARIMAX model
def fit_model(series, order, seasonal_order):
    model = SARIMAX(
    series,
    order = order,
    seasonal_order = seasonal_order,
    enforce_stationarity = False,
    enforce_invertibility = False
)
    
    return model.fit(disp = False)
    
#plot the forecast
def plot_forecast(df, forecast_mean, conf_int):
    fig = go.Figure()

    #historical data
    fig.add_trace(go.Scatter(
        x = df.index,
        y = df['RSXFS'],
        name = 'Historical',
        mode = 'lines',
        line = dict(color='steelblue')
    ))

    #forecast
    fig.add_trace(go.Scatter(
        x = forecast_mean.index,
        y = forecast_mean,
        name = 'Forecast',
        mode = 'lines',
        line = dict(color = 'orange', dash = 'dash')
    ))

    #confidence intervals
    fig.add_trace(go.Scatter(
        x = conf_int.index.tolist() + conf_int.index[::-1].tolist(),
        y = conf_int.iloc[:,0].tolist() + conf_int.iloc[:,1][::-1].tolist(),
        fill = 'toself',
        fillcolor = 'rgba(255,165,0,0.2)',
        line = dict(color='rgba(255,255,255,0)'),
        hoverinfo = 'skip',
        name = '95% Confidence Interval'
    ))

    fig.update_layout(
        title = 'Forecast vs Historical Sales',
        xaxis_title = 'Date',
        yaxis_title = 'Sales (Millions USD)',
        hovermode = 'x unified',
        template = 'plotly_white'
    )

    return fig

#download forecast in CSV format
def convert_df_to_csv(df):
    return df.to_csv(index = True).encode('utf-8')

# #load the trained data SARIMAX model
# @st.cache_resource
# def load_model():
#     with open('sarimax_model.pkl', 'rb') as f:
#         return pickle.load(f)

#main app
st.title('US Retail and Food Services Sales Forecast')
st.markdown('A time series forecasting tool with interactive controls and export features')

#load the data and model 
df = load_data()
# model = load_model()

#forecast horizon selector
st.sidebar.header('Forecast Settings')
steps = st.sidebar.slider('Forecast Horizon (Bi-Weekly Steps):', 4, 52, 12)

with st.sidebar.expander("Advanced Model Parameters"):
    p = st.number_input("AR order (p)", 0, 5, 2)
    d = st.number_input("Differencing order (d)", 0, 2, 1)
    q = st.number_input("MA order (q)", 0, 5, 2)
    P = st.number_input("Seasonal AR (P)", 0, 3, 1)
    D = st.number_input("Seasonal Diff (D)", 0, 2, 1)
    Q = st.number_input("Seasonal MA (Q)", 0, 3, 1)
    s = st.number_input("Seasonal Period (s)", 1, 52, 26)

#fit SARIMAX model
st.info('Fitting SARIMAX Model...This may take a while.')
result = fit_model(df['RSXFS'], (p,d,q), (P,D,Q,s))

#generate the forecast
forecast = result.get_forecast(steps=steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

#display interactive chart
st.plotly_chart(plot_forecast(df, forecast_mean, conf_int), use_container_width = True)

#display the last few rows
with st.expander('Forecast Data Table'):
    forecast_df = pd.DataFrame({
        'Forecast': forecast_mean,
        'Lower Confidence Interval': conf_int.iloc[:,0],
        'Upper Confidence Interval': conf_int.iloc[:,1]
    })

    st.dataframe(forecast_df.round(2))

#download the forecast
csv = convert_df_to_csv(forecast_df)
st.download_button(
    label = 'Download Forecast as CSV', 
    data = csv,
    file_name = 'forecast_output.csv',
    mime = 'text/csv'
)


