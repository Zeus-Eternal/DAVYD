import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from typing import Dict, Any, Optional
from dataclasses import dataclass
import base64
from io import BytesIO

@dataclass
class FigureData:
    """Container for visualization data with multiple output formats"""
    plotly_figure: Optional[Any] = None
    matplotlib_figure: Optional[Any] = None
    image_bytes: Optional[bytes] = None
    title: str = ""

class Visualization:
    """Enhanced visualization generator with multi-format support"""

    def __init__(self):
        self._figure_cache = {}

    def generate_dashboard(self, df: pd.DataFrame) -> Dict[str, FigureData]:
        figures = {}

        if "intent" in df.columns:
            figures['intent'] = self._create_pie_chart(df, 'intent', 'Intent Distribution')

        if "sentiment" in df.columns:
            figures['sentiment'] = self._create_bar_chart(df, 'sentiment', 'Sentiment Distribution')

        if "sentiment_polarity" in df.columns:
            figures['sentiment_polarity'] = self._create_histogram(df, 'sentiment_polarity', 'Sentiment Polarity Distribution', 20)

        if "tone" in df.columns:
            figures['tone'] = self._create_bar_chart(df, 'tone', 'Tone Distribution')

        if "category" in df.columns:
            figures['category'] = self._create_pie_chart(df, 'category', 'Category Distribution')

        if "keywords" in df.columns:
            figures['keywords'] = self._create_wordcloud(df['keywords'], 'Keywords Distribution')

        self._figure_cache = figures
        return figures

    def _create_pie_chart(self, df: pd.DataFrame, column: str, title: str) -> FigureData:
        counts = df[column].value_counts().reset_index()
        counts.columns = ['Label', 'Count']
        fig = px.pie(counts, names='Label', values='Count', title=title)
        return FigureData(plotly_figure=fig, title=title)

    def _create_bar_chart(self, df: pd.DataFrame, column: str, title: str) -> FigureData:
        counts = df[column].value_counts().reset_index()
        counts.columns = ['Label', 'Count']
        fig = px.bar(counts, x='Label', y='Count', title=title, color='Label')
        return FigureData(plotly_figure=fig, title=title)

    def _create_histogram(self, df: pd.DataFrame, column: str, title: str, bins: int) -> FigureData:
        fig = px.histogram(df, x=column, nbins=bins, title=title)
        return FigureData(plotly_figure=fig, title=title)

    def _create_wordcloud(self, series: pd.Series, title: str) -> FigureData:
        text = ' '.join(series.dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        img_bytes = self._fig_to_bytes(fig)
        plt.close(fig)

        return FigureData(
            matplotlib_figure=wordcloud,
            image_bytes=img_bytes,
            title=title
        )

    def _fig_to_bytes(self, fig) -> bytes:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        return buf.getvalue()

    def get_figure_as_html(self, figure_name: str) -> Optional[str]:
        if figure_name in self._figure_cache:
            fig_data = self._figure_cache[figure_name]
            if fig_data.plotly_figure:
                return fig_data.plotly_figure.to_html(full_html=False)
        return None

    def get_figure_as_image(self, figure_name: str) -> Optional[str]:
        if figure_name in self._figure_cache:
            fig_data = self._figure_cache[figure_name]
            if fig_data.plotly_figure:
                return fig_data.plotly_figure.to_image(format='png')
            elif fig_data.image_bytes:
                return base64.b64encode(fig_data.image_bytes).decode('utf-8')
        return None