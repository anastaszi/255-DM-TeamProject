import streamlit as st
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle as pk
import os
import warnings
warnings.filterwarnings("ignore")
# ml related
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
import warnings
from scipy import stats
from scipy.stats import norm, skew
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings('ignore')

train_path = "data/train.csv"
test_path = "data/test.csv"
real = "data/df_final.csv"
head = 'images/head.png'

@st.cache
def load_train_data(train_path):
    return pd.read_csv(train_path)

@st.cache
def load_test_data(test_path):
    return pd.read_csv(test_path)

@st.cache
def load_data(real):
    return pd.read_csv(real)


def save_data(value,res):
    file = 'db.csv'
    if not os.path.exists(file):
        with open(file,'w') as f:
            f.write("area, bedrooms, bathrooms\n")
    with open(file,'a') as f:
        data = f"{area},{bedrooms},{bathrooms},{res}\n"        
        f.write(data)

st.sidebar.image(head, caption="Project on California House price prediction !",use_column_width=True)




st.title("House Pricing Analysis in California")

menu=["About", "House Prediction","Predicted House","Visual"]
choices=st.sidebar.selectbox("Menu Bar",menu)

if choices=='House Prediction':
    categories={
        "House for sale": 0,
        "Condo for sale": 1,
        "Home for sale": 2,
        "Multi-family home for sale": 3,
        "Coming soon": 4,
        "Lot / Land for sale": 5,
        "New construction": 6,
        "Townhouse for sale": 7,
        "Foreclosure": 8
    }
    st.subheader("House Prediction in California")
    listingType=st.selectbox("Select Listing Type", list(categories.keys()), 0)
    listingTypeFinal=categories[listingType]
    area= st.number_input("Enter Area (in Sqft)",value=0,min_value=0,format='%i')
    bedrooms=st.number_input("Enter number of bedrooms",min_value=1,max_value=10,format='%i')
    bathrooms=st.number_input("Enter number of Bathrooms",min_value=1,max_value=10,format='%i')
    zipcode= st.number_input("Enter the zipcode",format='%f')
    submit = st.button('Predict')
    
    if submit:
        st.success("Prediction Done")
        value=[ bathrooms, 	bedrooms, listingTypeFinal, area,	zipcode]
        df=pd.DataFrame(value).transpose()
        
        model=pk.load(open('model/regressor1.pkl','rb'))
        ans=int(model.predict(df)) * 5
        st.subheader(f"The price is {ans} (USD) ")
        save_data(value,ans)

if choices=='Predicted House':
    st.subheader("Predicted House")
    st.info("expand to see data clearly")
    if os.path.exists("db.csv"):
        data = pd.read_csv('db.csv')
        st.write(data)
    else:
        st.error("please try some prediction, then the data will be available here")
if choices=='About':
    st.subheader("About Us")
    info='''
  Welcome to the California House price prediction application !
  We are a team of 4 and helping coustomers purchase their affordable dreamhouse in California State :)
  
  Summary:

A home means a future, stability. Every person's dream is to buy their dream house in a lifetime and sell it eventually with a good price and to find a better place as need changes.
Living a California dream is becoming more difficult each day. There are a lot of job opportunities out there with great weather, lifestyle and increasing demand for housing.
House prices are always changing and it is humanly impossible to determine what the price might be tomorrow. According to the state housing department, California needs to build 180,000 new houses every year in order to keep up with demand, which is very high, there is a big valley between supply and demand. It is very important to know where you invest your money and get maximum benefit. So with help of historical data and machine learning algorithms we are trying to help people to buy affordable houses.
In our project we are trying to find the future price of  the houses by analyzing past trends in California. We have made use of Random Forest regressor to predict the house prices based on the attributes like - area, number of bedrooms, number of bathrooms, zip code and listing type. This project is developed, and built using Streamlit and hosted on  AWS cloud.


    '''
    st.markdown(info,unsafe_allow_html=True)

if choices=='Visual':
    st.subheader("Data Visualization")      

    train_data = load_train_data(train_path)
    test_data = load_test_data(test_path)


    if st.checkbox("view dataset colum description"):
        st.subheader('displaying the column wise stats for the dataset')
        st.write(train_data.columns)
        st.write(train_data.describe())

    st.subheader('Correlation b/w dataset columns')
    corrmatrix = train_data.corr()
    f,ax = plt.subplots(figsize=(20,9))
    sns.heatmap(corrmatrix,vmax = .8, annot=True)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    

    st.subheader("most correlated features")
    top_corr = train_data.corr()
    top_corr_feat = corrmatrix.index[abs(corrmatrix['area_sqft'])>.5]
    plt.figure(figsize=(10,10))
    sns.heatmap(train_data[top_corr_feat].corr(), annot=True, cmap="RdYlGn")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)


    st.subheader("Comparing Overall Quality vs Sale Price")
    sns.barplot(train_data.area_sqft, train_data.price)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)


    st.subheader("Pairplot visualization to describe correlation easily")
    sns.set()
    cols = ['bathrooms', 'bedrooms', 'area_sqft', 'price']
    sns.pairplot(train_data[cols], size=2.5)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)


    st.subheader("Analyis of Sale Price column in dataset")
    sns.distplot(train_data['price'] , fit=norm)# Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train_data['price'])
    st.write( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    st.set_option('deprecation.showPyplotGlobalUse', False)


    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)


    fig = plt.figure(figsize=(10,10))
    res = stats.probplot(train_data['price'], plot=plt,)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
