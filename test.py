import streamlit as st
import pandas as pd
import numpy as np

# Set the title of the app
st.title('My First Streamlit App')

# Display some text
st.write('Hello, welcome to my app!')

# Create a simple dataframe
data = pd.DataFrame({
    'A': np.random.randn(10),
    'B': np.random.randn(10),
    'C': np.random.randn(10)
})

# Display the dataframe
st.write('Here is a simple dataframe:')
st.write(data)

# Display a line chart
st.write('And here is a line chart:')
st.line_chart(data)

# Add a slider
st.write('Use the slider to select a value:')
value = st.slider('Select a value', 0, 100, 50)

# Display the selected value
st.write(f'You selected: {value}') 