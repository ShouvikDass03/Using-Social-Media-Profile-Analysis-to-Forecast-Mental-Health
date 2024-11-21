import streamlit as st
import pickle
import string
import praw
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import random


nltk.download('punkt_tab')
nltk.download('stopwords')

client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = st.secrets["user_agent"]

# Initialize PorterStemmer
ps = PorterStemmer()


# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Reddit API credentials
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

st.title("Suicide Ideation Detection")

# Input area for manual text
input_sms = st.text_area("Enter the message manually:")

# Prediction for manual input
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)  # Preprocess
    vector_input = tfidf.transform([transformed_sms])  # Vectorize
    result = model.predict(vector_input.toarray())[0]  # Predict

    if result == 1:
        st.header("Suicidal Post")
    else:
        st.header("Non-suicidal Post")

# Input for subreddit name
subreddit_name = st.text_input("Enter subreddit name (default: 'random')", value='random')

# Fetch Reddit post and classify
if st.button('Fetch and Predict from Reddit'):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        # Try fetching a random post
        random_post = subreddit.random()

        if random_post is None:  # Handle None response from subreddit.random()
            # Fallback to fetching from 'hot'.
            st.warning(f"`random()` method failed for r/{subreddit_name}. Fetching from 'hot' posts.")
            posts = list(subreddit.hot(limit=10))  # Fetch 10 posts from 'hot'
            if posts:
                random_post = random.choice(posts)  # Randomly select one post
            else:
                st.error(f"No posts available in r/{subreddit_name}. Try another subreddit.")
                random_post = None

        if random_post:
            # Combine title and selftext for classification
            post_content = random_post.title + " " + random_post.selftext

            # Display post content, author, and subreddit
            st.write("### Reddit Post Details:")
            st.write(f"**Subreddit**: r/{random_post.subreddit}")
            st.write(f"**Author**: u/{random_post.author}")
            st.write(f"**Content**: {post_content}")

            # Preprocess and classify
            transformed_post = transform_text(post_content)
            vector_input = tfidf.transform([transformed_post])
            result = model.predict(vector_input.toarray())[0]

            if result == 1:
                st.header("Suicidal Post")
            else:
                st.header("Non-suicidal Post")
    except Exception as e:
        st.error(f"An error occurred: {e}")
