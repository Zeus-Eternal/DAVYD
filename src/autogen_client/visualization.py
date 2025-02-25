# autogen_client/visualization.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

class Visualization:
    """Class for generating data visualizations"""

    def generate_dashboard(self, df: pd.DataFrame):
        """Generate a dashboard with various charts"""
        st.subheader("📊 Data Quality Dashboard")

        # Intent Distribution (Pie Chart)
        if "intent" in df.columns:
            intent_counts = df['intent'].value_counts().reset_index()
            intent_counts.columns = ['Intent', 'Count']
            fig = px.pie(intent_counts, names='Intent', values='Count', title='Intent Distribution')
            st.plotly_chart(fig)

        # Sentiment Distribution (Bar Chart)
        if "sentiment" in df.columns:
            sentiment_counts = df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Distribution', color='Sentiment')
            st.plotly_chart(fig)

        # Sentiment Polarity Distribution (Histogram)
        if "sentiment_polarity" in df.columns:
            fig = px.histogram(df, x='sentiment_polarity', nbins=20, title='Sentiment Polarity Distribution')
            st.plotly_chart(fig)

        # Tone Distribution (Bar Chart)
        if "tone" in df.columns:
            tone_counts = df['tone'].value_counts().reset_index()
            tone_counts.columns = ['Tone', 'Count']
            fig = px.bar(tone_counts, x='Tone', y='Count', title='Tone Distribution', color='Tone')
            st.plotly_chart(fig)

        # Category Distribution (Pie Chart)
        if "category" in df.columns:
            category_counts = df['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            fig = px.pie(category_counts, names='Category', values='Count', title='Category Distribution')
            st.plotly_chart(fig)

        # Keywords Distribution (Word Cloud)
        if "keywords" in df.columns:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            keyword_text = ' '.join(df['keywords'].dropna().astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keyword_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)