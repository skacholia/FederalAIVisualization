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
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='Federal AI Inventory Analysis', page_icon='📊')
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

@st.cache_data
def load_data():
    df = pd.read_csv('federalai_embed.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    df['cluster'] = df['cluster'] + 1
    return df

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

def gpt(prompt, text, engine="gpt-3.5-turbo-1106", temperature=0.2):
    stream = client.chat.completions.create(
            model=engine,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            stream=True,
            temperature=temperature
    )
    return stream

st.title('Federal AI Inventory Analysis')
content = """[Executive Order 13960](https://www.cio.gov/policies-and-priorities/Executive-Order-13960-AI-Use-Case-Inventories-Reference/), “Promoting the Use of Trustworthy Artificial Intelligence in the Federal Government,” 
requires US federal agencies to prepare an inventory of non-classified and non-sensitive current and  planned Artificial Intelligence (AI) use cases. 
This tool helps navigate, understand, and visualize those use cases. Using the sidebar, you can break down the data by department.\n\n
I owe gratitude to [Travis Hoppe](https://github.com/thoppe/Federal-AI-inventory-analysis-2023), for cleaning use case inventory data. I incorporate two main features in this app:
\n\n

**1. Search**: I used OpenAI's `text-embedding-3-small` model to embed the descriptions of use cases. You can enter a query, 
and the tool will find the most similar use cases to that query. **This is based on meaning, not specific wording or letters.** 
For example, 'school' and 'education' will have similar embeddings, 
despite having few letters in common. \nI also use `gpt-3.5-turbo` to generate short summaries of the found use cases. 
**This could help AI practitioners take inspiration from past AI projects.**\n


**2. Visualization**: I've included an interactive, 3D visualization of the embeddings of the project descriptions. 
This helps visually identify which projects are similar to which. I've also included graphical breakdowns of stages of projects and AI techniques used."""

df = load_data()
department_list = ['Overall'] + list(df['Department'].unique())
selected_department = st.sidebar.selectbox('Select a Department:', department_list)
if selected_department == 'Overall':
    df = df
    st.header("Context")
    st.markdown(content)
else:
    df = df[df['Department'] == selected_department]

st.sidebar.markdown("This is project by [Suhan Kacholia](https://skacholia.vercel.app/)")
st.header("Search")
search_query = st.text_input('Enter your search query:', '')
if st.button('Search'):
    if search_query:
        # Perform the search
        results = search(df, search_query, n=10, pprint=False)
        results = results[['Title', 'Summary',  'keywords', 'cluster', 'Use_Case_ID', 'Department','Department_Code', 'Agency', 'Office', 'Development_Stage', 'Techniques']]
        st.write("Search Results:")
        st.dataframe(results)  # This will display the DataFrame in the app
        summary_string = results['Summary'].str.cat(sep=' ')
        st.markdown("### GPT Summary:")
        st.write_stream(gpt("Provide a one-paragraph summary of these AI projects. Mention specifics.", summary_string))

st.header("3D Visualization")
st.markdown("""This is an interactive visualization of the embeddings of project descriptions. You can see which projects are close to each other, based on the meaning of their description. I've also clustered the project description, finding 5 natural thematic clusters:

**1. Image Detection:** These projects focus on the analysis of sensors and imagery data. Common keywords include **satellite imagery**, **sensor data**, **lidar imagery**, and **computer vision**. For example the DHS leverages SAR satellites and AI for detection capabilities, and the NOAA uses AI for processing passive acoustic data of whales.

**2. Document Processing:** These projects use natural language processing (NLP) and other text analysis techniques for tasks like claims processing, creating chatbots, and processing grants. Common keywords include **text analytics** and **virtual agent**. For example, the FDA uses AI to process docket comments.

**3. Trend Forecasting:**: These projects use AI for predictive analytics and forecasting. Common keywords include **predictive scenarios** and **seasonal forecasts**. For example, the DOI uses AI to forecast water supply.

**4. Energy and Physics:** These projects focus on using AI for projects in energy and physics. Common keywords include **energy storage**, **boiler operations**, and **combustion**. For example, the National Energy Technology Laboratory is building an artificial intelligence field laboratory for unconventional energy resources.

**5. Health:** These projects focus on drug discovery, clinical applications, and other medical uses of AI. Common keywords include **claims data**, **drug labeling**, and **opioid use**. For example, the CDC is using AI to detect illicit opioid use based on clinical notes.""")
embeddings = df['embedding'].tolist()
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

st.plotly_chart(figure(df), use_container_width=True)

st.header("More Visualizations")
col1, col2 = st.columns(2)

stage_counts = df['Development_Stage'].value_counts().reset_index()
stage_counts.columns = ['Stage', 'Count']
title_text = 'Stages of Implementation'
pie_fig = px.pie(stage_counts, names='Stage', values='Count', title=title_text)
st.plotly_chart(pie_fig, use_container_width=True)

with col1:
    if selected_department == 'Overall':
        agency_counts = df['Department'].value_counts().reset_index()
        agency_counts.columns = ['Department', 'Count']
        title_text = 'Projects by Department'
        top_agency_counts = agency_counts.head(5)
        bar_fig = px.bar(top_agency_counts, x='Department', y='Count', title='Top 5 Departments')
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        agency_counts = df['Agency'].value_counts().reset_index()
        agency_counts.columns = ['Agency', 'Count']
        title_text = 'Projects by Agency'
        top_agency_counts = agency_counts.head(5)
        bar_fig = px.bar(top_agency_counts, x='Agency', y='Count', title='Top 5 Agencies')
        st.plotly_chart(bar_fig, use_container_width=True)

with col2:
    techniques_series = df['Techniques'].str.split(', ')
    exploded_techniques = techniques_series.explode()
    technique_counts = exploded_techniques.value_counts().reset_index()
    technique_counts.columns = ['Technique', 'Count']
    top_technique_counts = technique_counts.head(5)
    bar_fig = px.bar(top_technique_counts, x='Technique', y='Count', title='Top 5 Techniques')
    st.plotly_chart(bar_fig, use_container_width=True)