import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import csv

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Convert the uploaded file to a list of lists
    csv_reader = csv.reader(uploaded_file.read().decode('utf-8').splitlines())
    dataset = list(csv_reader)
    st.write(dataset)

    # Preprocessing
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    st.write(df)

    # User input for min_support
    min_support = st.slider('Select min_support value', min_value=0.0, max_value=1.0, step=0.01)

    # Apply fpgrowth
    result = fpgrowth(df, min_support=min_support, use_colnames=True)
    result['itemsets'] = result['itemsets'].apply(lambda x: ', '.join(list(x)))
    st.table(result)

# open the terminal where u have installed the streamlit module and run the code and u get a url for the code to run either
#paste it in browser or wait for it to automatically open in the browser.