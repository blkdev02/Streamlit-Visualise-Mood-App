import streamlit as st
import glob, nltk
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer


# Get all file paths
filepaths = sorted(glob.glob("diary/*.txt"))

# nltk
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

negativity = []
positivity = []
for filepath in filepaths:
    with open(filepath) as file:
        content = file.read()
    scores = analyzer.polarity_scores(content)
    negativity.append(scores["neg"])
    positivity.append(scores["pos"])

dates = [name.strip(".txt").strip("diary/").strip("\\") for name in filepaths]

# Web app
st.title("Diary Tone")

st.subheader("Positivity")
pos_figure = px.line(x=dates, y= positivity, labels={"x": "Dates", "y": "Positivity"})
st.plotly_chart(pos_figure)

st.subheader("Negativity")
neg_figure = px.line(x=dates, y= negativity, labels={"x": "Dates", "y": "Negativity"})
st.plotly_chart(neg_figure)



