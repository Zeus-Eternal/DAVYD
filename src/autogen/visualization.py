# src/autogen/visualization.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class Visualization:
    def __init__(self):
        logging.info("Visualization module initialized.")
    
    def plot_sentiment_distribution(self, df: pd.DataFrame):
        if 'sentiment' in df.columns:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.warning("No 'sentiment' column found for visualization.")
    
    def plot_intent_distribution(self, df: pd.DataFrame):
        if 'intent' in df.columns:
            st.subheader("Intent Distribution")
            intent_counts = df['intent'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=intent_counts.index, y=intent_counts.values, ax=ax)
            ax.set_xlabel('Intent')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.warning("No 'intent' column found for visualization.")
    
    def plot_sentiment_polarity_statistics(self, df: pd.DataFrame):
        if 'sentiment_polarity' in df.columns:
            st.subheader("Sentiment Polarity Statistics")
            polarity_mean = pd.to_numeric(df['sentiment_polarity'], errors='coerce').mean()
            polarity_median = pd.to_numeric(df['sentiment_polarity'], errors='coerce').median()
            st.write(f"**Mean Sentiment Polarity:** {polarity_mean:.2f}")
            st.write(f"**Median Sentiment Polarity:** {polarity_median:.2f}")
        else:
            st.warning("No 'sentiment_polarity' column found for statistics.")
    
    def plot_keyword_frequency(self, df: pd.DataFrame):
        if 'keywords' in df.columns:
            st.subheader("Keyword Frequency")
            # Split keywords by quotes and spaces
            keywords_series = df['keywords'].str.split('" "').explode().str.strip('"')
            keyword_counts = keywords_series.value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=keyword_counts.values, y=keyword_counts.index, ax=ax)
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Keywords')
            st.pyplot(fig)
        else:
            st.warning("No 'keywords' column found for visualization.")
    
    def generate_dashboard(self, df: pd.DataFrame):
        """
        Generate all visualizations in a dashboard layout.
        
        :param df: The dataset as a pandas DataFrame.
        """
        self.plot_sentiment_distribution(df)
        self.plot_intent_distribution(df)
        self.plot_sentiment_polarity_statistics(df)
        self.plot_keyword_frequency(df)
