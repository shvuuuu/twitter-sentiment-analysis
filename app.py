import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tweepy
from plotly.subplots import make_subplots
from transformers import pipeline

consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_key,access_secret)
api = tweepy.API(auth)


def get_tweets(username, count):
    tweets = tweepy.Cursor(
        api.user_timeline,
        screen_name=username,
        tweet_mode="extended",
        exclude_replies=True,
        include_rts=False,
    ).items(count)

    tweets = list(tweets)
    response = {
        "tweets": [tweet.full_text.replace("\n", "").lower() for tweet in tweets],
        "timestamps": [str(tweet.created_at) for tweet in tweets],
        "retweets": [tweet.retweet_count for tweet in tweets],
        "likes": [tweet.favorite_count for tweet in tweets],
    }
    return response


def get_sentiment(texts):
    preds = pipe(texts)

    response = dict()
    response["labels"] = [pred["label"] for pred in preds]
    response["scores"] = [pred["score"] for pred in preds]
    return response


def neutralise_sentiment(preds):
    for i, (label, score) in enumerate(zip(preds["labels"], preds["scores"])):
        if score < 0.5:
            preds["labels"][i] = "neutral"
            preds["scores"][i] = 1.0 - score


def get_aggregation_period(df):
    t_min, t_max = df["timestamps"].min(), df["timestamps"].max()
    t_delta = t_max - t_min
    if t_delta < pd.to_timedelta("30D"):
        return "1D"
    elif t_delta < pd.to_timedelta("365D"):
        return "7D"
    else:
        return "30D"


@st.cache_data
def load_model():
    pipe = pipeline(task="sentiment-analysis", model="bhadresh-savani/distilbert-base-uncased-emotion")
    return pipe


"""
# Twitter Emotion Analyser
"""


pipe = load_model()
twitter_handle = st.sidebar.text_input("Twitter handle:", "elonmusk")
twitter_count = st.sidebar.selectbox("Number of tweets:", (10, 30, 50, 100))


if st.sidebar.button("Get tweets!"):
    tweets = get_tweets(twitter_handle, twitter_count)
    preds = get_sentiment(tweets["tweets"])
    # neutralise_sentiment(preds)
    tweets.update(preds)
    # dataframe creation + preprocessing
    df = pd.DataFrame(tweets)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    # plots
    agg_period = get_aggregation_period(df)
    ts_sentiment = (
        df.groupby(["timestamps", "labels"])
        .count()["likes"]
        .unstack()
        .resample(agg_period)
        .count()
        .stack()
        .reset_index()
    )
    ts_sentiment.columns = ["timestamp", "label", "count"]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.15)

    # TODO: check that stacking makes sense!
    for label in ts_sentiment["label"].unique():
        fig.add_trace(
            go.Scatter(
                x=ts_sentiment.query("label == @label")["timestamp"],
                y=ts_sentiment.query("label == @label")["count"],
                mode="lines",
                name=label,
                stackgroup="one",
                hoverinfo="x+y",
            ),
            row=1,
            col=1,
        )

    likes_per_label = df.groupby("labels")["likes"].mean().reset_index()

    fig.add_trace(
        go.Bar(
            x=likes_per_label["labels"],
            y=likes_per_label["likes"],
            showlegend=False,
            marker_color=px.colors.qualitative.Plotly,
            opacity=0.6,
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="Number of Tweets", row=1, col=1)
    fig.update_yaxes(title_text="Number of Likes", row=1, col=2)
    fig.update_layout(height=350, width=750)

    st.plotly_chart(fig)

    # tweet sample
    st.markdown(df.to_markdown())
