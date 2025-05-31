import streamlit as st
import pickle
import string
import praw
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import random

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Load secrets
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = st.secrets["user_agent"]

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# NLP setup
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# PRAW setup
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# UI
st.title("Social Media Profile for Predicting the Psychological State")

# Input
subreddit_name = st.text_input("Enter subreddit name", value='AskReddit')

if st.button('Fetch and Predict from Reddit'):
    subreddit_name = subreddit_name.strip().replace(" ", "")
    try:
        subreddit = reddit.subreddit(subreddit_name)
        subreddit.id  # Validate subreddit exists

        # Try fetching a random post
        random_post = None
        try:
            random_post = subreddit.random()
        except:
            st.warning(f"`random()` failed for r/{subreddit_name}. Falling back to hot posts.")

        # If random fails or returns None
        if random_post is None:
            posts = list(subreddit.hot(limit=10))
            if posts:
                random_post = random.choice(posts)
            else:
                st.error(f"No posts found in r/{subreddit_name}. Try another subreddit.")
                st.stop()

        # Extract and display post
        post_content = random_post.title + " " + random_post.selftext
        post_url = f"https://www.reddit.com{random_post.permalink}"
        st.write("### Reddit Post Details:")
        st.write(f"**Subreddit**: r/{random_post.subreddit}")
        st.write(f"**Author**: u/{random_post.author}")
        st.write(f"**Content**: {post_content}")
        st.write(f"**Link**: [View Post]({post_url})")

        # Predict
        transformed_post = transform_text(post_content)
        vector_input = tfidf.transform([transformed_post])
        result = model.predict(vector_input.toarray())[0]

        st.header("🧠 Suicidal Post" if result == 1 else "✅ Non-suicidal Post")

    except praw.exceptions.RedditAPIException as e:
        st.error(f"Reddit API error: {e}")
    except Exception as e:
        if '400' in str(e):
            st.error("Bad request. Subreddit might not exist or contains invalid characters.")
        elif '404' in str(e):
            st.error("Subreddit not found. Please enter a valid subreddit name.")
        else:
            st.error(f"An unexpected error occurred: {e}")
