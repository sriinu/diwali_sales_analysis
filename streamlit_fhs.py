import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.write("# Sale Prediction")

col1, col2, col3 = st.columns(3)

# getting user inputgender = col1.selectbox("Enter your gender",["Male", "Female"])

maps = {'Gender': {'F': 1, 'M': 2}, 
        
        'Age Group': {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}, 
        
        'Marital_Status': {0: 1, 1: 2}, 
        
        'State': {'Andhra Pradesh': 1, 'Bihar': 2, 'Delhi': 3, 'Gujarat': 4, 'Haryana': 5, 'Himachal Pradesh': 6, 
                  'Jharkhand': 7, 'Karnataka': 8, 'Kerala': 9, 'Madhya Pradesh': 10, 'Maharashtra': 11, 'Punjab': 12, 
                  'Rajasthan': 13, 'Telangana': 14, 'Uttar Pradesh': 15, 'Uttarakhand': 16}, 
                  
        'Zone': {'Central': 1, 'Eastern': 2, 'Northern': 3, 'Southern': 4, 'Western': 5}, 
                
        'Occupation': {'Agriculture': 1, 'Automobile': 2, 'Aviation': 3, 'Banking': 4, 'Chemical': 5, 
                                 'Construction': 6, 'Food Processing': 7, 'Govt': 8, 'Healthcare': 9, 'Hospitality': 10, 
                                 'IT Sector': 11, 'Lawyer': 12, 'Media': 13, 'Retail': 14, 'Textile': 15}, 
        'Product_Category': {'Auto': 1, 'Beauty': 2, 'Books': 3, 'Clothing & Apparel': 4, 'Decor': 5, 
                                        'Electronics & Gadgets': 6, 'Food': 7, 'Footwear & Shoes': 8, 'Furniture': 9, 
                                        'Games & Toys': 10, 'Hand & Power Tools': 11, 'Household items': 12, 'Office': 13, 
                                        'Pet Care': 14, 'Sports Products': 15, 'Stationery': 16, 'Tupperware': 17, 'Veterinary': 18}}

with col1:
    gender = st.selectbox("Gender", ["M", "F"])

    state = st.selectbox("State",['Maharashtra', 'Andhra Pradesh', 'Uttar Pradesh', 'Karnataka', 'Gujarat',
                                'Himachal Pradesh', 'Delhi', 'Madhya Pradesh', 'Jharkhand', 'Kerala',
                                'Haryana', 'Bihar', 'Rajasthan', 'Uttarakhand', 'Telangana', 'Punjab'])
    
    zone = st.selectbox("Zone", ["Northern","Eastern","Western","Central","Southern"])


with col2:
    age_group = st.selectbox("Age Group", ["0-17","18-25","26-35","36-45","46-50","51-55","55+"])
    
    marital_status = st.radio("Marital Status", ["Married", "Unmarried"])
    


with col3:
    occupation = st.selectbox("Occupation",['Healthcare', 'Govt', 'Automobile', 'Construction', 'Food Processing',
                                                'Lawyer', 'IT Sector', 'Media', 'Banking', 'Retail', 'Hospitality', 'Aviation',
                                                'Agriculture', 'Textile', 'Chemical'])
    
    product_category = st.selectbox("Product_Category",['Auto','Hand & Power Tools','Stationery','Tupperware','Footwear & Shoes',
                                                        'Furniture', 'Food', 'Games & Toys', 'Sports Products', 'Books',
                                                        'Electronics & Gadgets', 'Decor', 'Clothing & Apparel', 'Beauty',
                                                        'Household items', 'Pet Care', 'Veterinary', 'Office'])
    
  

df_pred = pd.DataFrame(
        [[gender, age_group, marital_status, state, zone, occupation, product_category]],
        columns=['Gender', 'Age Group', 'Marital_Status', 'State', 'Zone', 'Occupation', 'Product_Category']
    )

df_pred['Marital_Status'] = df_pred['Marital_Status'].apply(lambda x: 1 if x == 'Married' else 0)


for col in df_pred.columns:
    try:
        df_pred[col] = maps[col][df_pred[col].iloc[0]]
    except:
        continue

model = joblib.load('final.joblib')
print(df_pred)
prediction = model.predict(df_pred)

if st.button('Predict'):
    st.write('Estimated Amount: ',prediction)
