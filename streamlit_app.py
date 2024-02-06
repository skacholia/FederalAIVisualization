import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import ast
from openai import OpenAI
import openai
from sklearn.manifold import TSNE
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='AI Implementation Stages and Techniques by Agency', page_icon='📊')

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
client = OpenAI()

def get_embedding(text):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search(df, text, n=3, pprint=True):
    embedding = np.array(get_embedding(text)).reshape(1, -1)
    df['similarity'] = df.embedding.apply(lambda x: cosine_similarity(np.array(x).reshape(1, -1), embedding))
    res = df.sort_values('similarity', ascending=False).head(n)
    return res

def sim(text, target):
    embedding = get_embedding(text)
    return cosine_similarity(embedding, target)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('federalai_embed.csv')
    return df  # Update this path

df = load_data()

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


search_query = st.text_input('Enter your search query:', '')
if st.button('Search'):
    if search_query:
        # Perform the search
        results = search(df, search_query, n=10, pprint=False)
        st.write("Search Results:")
        st.dataframe(results)  # This will display the DataFrame in the app

embeddings = filtered_df['embedding'].tolist()
if isinstance(embeddings[0], str):
    embeddings = [ast.literal_eval(e) for e in embeddings]
tsne_embeddings = TSNE(n_components=3, random_state=42).fit_transform(np.array(embeddings))
x = tsne_embeddings[:, 0]
y = tsne_embeddings[:, 1]
z = tsne_embeddings[:, 2]

@st.cache_data
def figure(df):
    scatter = go.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode = 'markers',
        marker = dict(
            size = 5,  # Increase the marker size for better visibility
            color = df['cluster'],  # Color by cluster
            colorscale = 'Viridis',
            opacity = 0.8,
        ),
        hovertemplate = 
            '%{text}<br><br>' +  # Bold name on hover
            '<b>Cluster: %{customdata}<br>' +  # Include cluster information
            'Coordinates: (%{x}, %{y}, %{z})<extra></extra>',  # Include coordinates
        text = ['<br>'.join(text[i:i+30] for i in range(0, len(text), 20)) for text in df['Title']],
        customdata = df['cluster']
    )

    layout = go.Layout(
        title = '3D t-SNE Clustering',
        scene = dict(
            xaxis = dict(title='Dimension 1', zeroline=False),
            yaxis = dict(title='Dimension 2', zeroline=False),
            zaxis = dict(title='Dimension 3', zeroline=False),
        ),
        hoverlabel = dict(
            bgcolor = "black",  # Background color for hover label
            font_size = 12,  # Text font size
            font_family = "Arial"  # Text font family
        )
    )

    fig = go.Figure(data=[scatter], layout=layout)
    return fig

st.plotly_chart(figure(filtered_df), use_container_width=True)

# Use columns to layout the pie and bar charts side by side
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(pie_fig, use_container_width=True)

with col2:
    st.plotly_chart(bar_fig, use_container_width=True)