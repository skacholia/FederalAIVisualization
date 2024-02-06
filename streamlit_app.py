import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title='AI Implementation Stages and Techniques by Agency', page_icon='📊')


# Load the dataset
df = pd.read_csv('federalai.csv')  # Update this path

# Sidebar for agency selection, adding an "Overall" option
department_list = ['Overall'] + list(df['Department'].unique())
selected_department = st.sidebar.selectbox('Select a Department:', department_list)

# Adjust the dataset filtering
if selected_department == 'Overall':
    filtered_df = df  # Use the entire dataset for the "Overall" option
else:
    filtered_df = df[df['Department'] == selected_department]  # Filter by the selected agency

# Count the occurrences of each development stage for the selected dataset
stage_counts = filtered_df['Development_Stage'].value_counts().reset_index()
stage_counts.columns = ['Stage', 'Count']

# Create the pie chart for the selected dataset
title_text = 'Stages of Implementation'
pie_fig = px.pie(stage_counts, names='Stage', values='Count', title=title_text)

# Count and plot the occurrences of techniques
techniques_series = filtered_df['Techniques'].str.split(', ')
exploded_techniques = techniques_series.explode()
technique_counts = exploded_techniques.value_counts().reset_index()
technique_counts.columns = ['Technique', 'Count']
top_technique_counts = technique_counts.head(5)

# Create the Plotly bar graph for techniques
bar_fig = px.bar(top_technique_counts, x='Technique', y='Count', title='Top 5 Techniques')

# Streamlit app code to display the charts side by side
st.title('AI Implementation Stages and Techniques Visualization')

# Use columns to layout the pie and bar charts side by side
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(pie_fig, use_container_width=True)

with col2:
    st.plotly_chart(bar_fig, use_container_width=True)