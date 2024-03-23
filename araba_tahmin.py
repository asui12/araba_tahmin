#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder 
import streamlit as st


# In[13]:


df=pd.read_excel("cars.xls")


# In[14]:


#df.to_csv('cars.csv',index=False) xls i csv ye cevirdik


# In[15]:


df.head()


# In[16]:


X=df.drop("Price",axis=1)
y=df[["Price"]]


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(X,y,
                                              test_size=0.2,
                                              random_state=42)


# In[18]:


preproccer=ColumnTransformer(transformers=[('num',StandardScaler(),
                                          ['Mileage','Cylinder','Liter','Doors']),
                                          
                                          ('cat',OneHotEncoder(),['Make','Model','Trim', 'Type'])])


# In[19]:


model=LinearRegression()
pipe=Pipeline(steps=[('preprocessor',preproccer),
                    ('model',model)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)



# In[21]:


def price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather):
	input_data=pd.DataFrame({
		'Make':[make],
		'Model':[model],
		'Trim':[trim],
		'Mileage':[mileage],
		'Type':[car_type],
		'Car_type':[car_type],
		'Cylinder':[cylinder],
		'Liter':[liter],
		'Doors':[doors],
		'Cruise':[cruise],
		'Sound':[sound],
		'Leather':[leather]
		})
	prediction=Pipeline.predict(input_data)[0]
	return prediction
st.title("Car Price Prediction :red_car: @drmurataltun")
st.write("Enter Car Details to predict the price of the car")
make=st.selectbox("Make",df['Make'].unique())
model=st.selectbox("Model",df[df['Make']==make]['Model'].unique())
trim=st.selectbox("Trim",df[(df['Make']==make) & (df['Model']==model)]['Trim'].unique())
mileage=st.number_input("Mileage",200,60000)
car_type=st.selectbox("Type",df['Type'].unique())
cylinder=st.selectbox("Cylinder",df['Cylinder'].unique())
liter=st.number_input("Liter",1,6)
doors=st.selectbox("Doors",df['Doors'].unique())
cruise=st.radio("Cruise",[True,False])
sound=st.radio("Sound",[True,False])
leather=st.radio("Leather",[True,False])
if st.button("Predict"):
	pred=price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather)

	st.write("Predicted Price :red_car:  $",round(pred[0],2))


# In[ ]:




