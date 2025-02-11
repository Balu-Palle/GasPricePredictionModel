import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('Nat_Gas.csv')
df['Dates']=pd.to_datetime(df['Dates'],format='%m/%d/%y',errors='coerce')
df['Dates_odinal']=df['Dates'].map(pd.Timestamp.to_julian_date)


x_train, x_test, y_train, y_test = train_test_split(df[['Dates_odinal']],df[['Prices']],test_size=0.2)
model=LinearRegression()
model.fit(x_test,y_test)

input_date=input('Enter the date in the format dd/mm/yyyy: ')
input_date=pd.to_datetime(input_date,format='%m/%d/%Y',errors='coerce')

if pd.isna(input_date):
    print("❌ Invalid date format. Please enter a valid date in mm/dd/yy or mm/dd/yyyy format.")
else:
    input_date_ordinal=pd.Timestamp(input_date).to_julian_date()
    input_date_ordinal = pd.DataFrame([[input_date_ordinal]], columns=['Dates_odinal'])
    price=model.predict(input_date_ordinal)
    print(price)
    print(f"The price of natural gas on {input_date_ordinal} is {price[0][0]:.2f}")

