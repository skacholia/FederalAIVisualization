import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('federalai.csv')  # Update this path

# Sidebar for agency selection, adding an "Overall" option
agency_list = ['Overall'] + list(df['Agency'].unique())
selected_agency = st.sidebar.selectbox('Select an Agency:', agency_list)

# Adjust the dataset filtering
if selected_agency == 'Overall':
    filtered_df = df  # Use the entire dataset for the "Overall" option
else:
    filtered_df = df[df['Agency'] == selected_agency]  # Filter by the selected agency

# Count the occurrences of each development stage for the selected dataset
stage_counts = filtered_df['Development_Stage'].value_counts().reset_index()
stage_counts.columns = ['Stage', 'Count']

# Create the pie chart for the selected dataset
title_text = 'Stages of Implementation - Overall' if selected_agency == 'Overall' else f'Stages of Implementation for {selected_agency}'
fig = px.pie(stage_counts, names='Stage', values='Count', title=title_text)

# Streamlit app code to display the pie chart
st.set_page_config(page_title='AI Implementation Stages by Agency', page_icon='📊')
st.title(title_text)
st.plotly_chart(fig, use_container_width=True)
