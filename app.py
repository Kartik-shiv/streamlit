import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("multiplekaggledata.csv")

dummy=pd.get_dummies(data,columns=["Neighborhood"])
dummy_val=dummy.astype(int)
# print(dummy_val)



x=dummy_val.drop(["Price"],axis='columns')
y=data['Price']

x_train,x_test,y_train,y_test=train_test_split(dummy_val[["SquareFeet","Bedrooms","Bathrooms","YearBuilt","Neighborhood_Rural","Neighborhood_Suburb","Neighborhood_Urban"]],data.Price,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)

st.sidebar.title("Input Features")
SquareFeet = st.sidebar.slider("choose no.of.squarefeet",1000,3000)
Bedrooms = st.sidebar.slider("choose no.of.Bedrooms",2,5)
Bathrooms = st.sidebar.slider("choose no.of.Bathrooms",1,3)
Neighborhood = ["Rural","Urban","Suburb"]
Neighborhood_input = st.sidebar.selectbox("Select the area where you want to buy the house",Neighborhood)
YearBuilt = st.sidebar.slider("Choose the year the house is Built:",1950,2021)

if Neighborhood_input == "Rural":
    user_data = [SquareFeet, Bedrooms, Bathrooms, YearBuilt, 1, 0, 0]
    st.image("rural.jpg")
elif Neighborhood_input == "Suburb":
    user_data = [SquareFeet, Bedrooms, Bathrooms, YearBuilt, 0, 1, 0]
    st.image('Suburb.jpg')
elif Neighborhood_input == "Urban":
    user_data = [SquareFeet, Bedrooms, Bathrooms, YearBuilt, 0, 0, 1]
    st.image('Urban.jpg')
else:
    print("Neighborhood not found")
    exit()

user_data = np.array(user_data).reshape(1, -1)  # Reshape to a 2D array for model prediction
predicted_price = model.predict(user_data)
val = int(predicted_price[0])
accuracy = model.score(x_test,y_test)
st.write(f"The price of the home is : {val}",)
st.write(f"The accuracy of the model is: {accuracy:.2f}")



