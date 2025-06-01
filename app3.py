import streamlit as st
import praw
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer, util
import pickle
import random
import requests
from PIL import Image
from io import BytesIO

# Downloads and Setup
nltk.download('stopwords')
nltk.download('punkt_tab')

# Secrets
device = "cpu"
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
user_agent = st.secrets["user_agent"]

# Load Models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# NLP Setup
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# Reddit and Embedding Setup
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Page Configuration
st.set_page_config(
    page_title="Mental Health Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {padding: 2rem;}
        .block {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        .subtitle {
            color: #34495e;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .description {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .warning {
            background-color: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .result-box {
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .profile-image {
            border-radius: 50%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .metric-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Main Layout
st.markdown('<h1 class="title">Mental Health Analysis System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyzing Social Media Content for Mental Health Indicators</p>', unsafe_allow_html=True)

# Create three columns for the main layout
left_col, middle_col, right_col = st.columns([1, 2, 1])

# Left Column - Project Description and Disclaimers
with left_col:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.markdown("### About the Project")
    st.markdown("""
    This system uses advanced machine learning algorithms to analyze social media content for potential mental health indicators. It provides two main functionalities:
    
    1. **Subreddit Post Analysis**: Analyzes random posts from specified subreddits for potential suicide risk indicators.
    
    2. **User BDI-II Assessment**: Estimates a user's BDI-II (Beck Depression Inventory-II) score based on their Reddit post history.
    """)
    
    st.markdown("### Important Disclaimers")
    st.markdown("""
    <div class="warning">
    ⚠️ This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
    
    🏥 If you or someone you know is experiencing a mental health crisis:
    - Call 9152987821 (National Suicide & Crisis Lifeline)
    - Seek immediate professional help
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Middle Column - Search Functionalities
with middle_col:
    # Subreddit Analysis Section
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("🔍 Subreddit Post Analysis")
    subreddit_name = st.text_input("Enter subreddit name", value='AskReddit').strip().replace(" ", "")
    
    if st.button('Analyze Random Post'):
        try:
            subreddit = reddit.subreddit(subreddit_name)
            _ = subreddit.id
            random_post = None
            try:
                random_post = subreddit.random()
            except:
                pass

            if random_post is None:
                posts = list(subreddit.hot(limit=20))
                if posts:
                    random_post = random.choice(posts)
                else:
                    st.error(f"No posts found in r/{subreddit_name}")
                    st.stop()

            post_content = random_post.title + " " + random_post.selftext
            
            # Store post details for right column
            st.session_state.current_post = {
                'subreddit': str(random_post.subreddit),
                'author': str(random_post.author),
                'content': post_content,
                'url': f"https://www.reddit.com{random_post.permalink}"
            }

            # Analysis
            transformed_post = transform_text(post_content)
            vector_input = tfidf.transform([transformed_post])
            result = model.predict(vector_input.toarray())[0]
            
            result_color = "red" if result == 1 else "green"
            result_text = "🚨 High Risk Indicators Detected" if result == 1 else "✅ No Significant Risk Detected"
            
            st.markdown(f"""
            <div class="result-box" style="background-color: {'#ffe6e6' if result == 1 else '#e6ffe6'}">
                <h3 style="color: {result_color}; text-align: center;">{result_text}</h3>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # BDI-II Analysis Section
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("📊 User BDI-II Analysis")
    reddit_user = st.text_input("Enter Reddit username")
    
    if st.button("Generate BDI-II Assessment"):
        try:
            redditor = reddit.redditor(reddit_user)
            posts = [submission.title + " " + submission.selftext for submission in redditor.submissions.new(limit=25)]
            
            if not posts:
                st.error("No posts found for this user.")
                st.stop()

            # Store user info for right column
            st.session_state.current_user = reddit_user

            post_embeddings = embedder.encode(posts, convert_to_tensor=True)
            bdi_score = 0
            bdi_breakdown = []

            for question, options in bdi_questions:
                option_embeddings = embedder.encode(options, convert_to_tensor=True)
                max_similarity = [0, 0, 0, 0]
                for post_embed in post_embeddings:
                    sims = util.cos_sim(post_embed, option_embeddings)[0]
                    for i in range(4):
                        max_similarity[i] = max(max_similarity[i], sims[i].item())
                score = max(range(4), key=lambda i: max_similarity[i])
                bdi_score += score
                bdi_breakdown.append((question, score))

            # Determine severity and color
            if bdi_score <= 13:
                severity = "Minimal Depression Indicators"
                color = "#28a745"
            elif bdi_score <= 19:
                severity = "Mild Depression Indicators"
                color = "#ffc107"
            elif bdi_score <= 28:
                severity = "Moderate Depression Indicators"
                color = "#fd7e14"
            else:
                severity = "Severe Depression Indicators"
                color = "#dc3545"

            score_percent = int((bdi_score / 63) * 100)
            
            st.markdown(f"""
            <div class="metric-box">
                <h4>BDI-II Score Assessment</h4>
                <div style="background-color: #e9ecef; border-radius: 8px; padding: 3px; margin: 10px 0;">
                    <div style="width: {score_percent}%; background-color: {color}; height: 24px; border-radius: 5px; transition: width 0.5s ease-in-out;"></div>
                </div>
                <h5>{severity}</h5>
                <p>Score: {bdi_score} / 63</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View Detailed Breakdown"):
                for question, score in bdi_breakdown:
                    if score == 0: c = "#28a745"
                    elif score == 1: c = "#ffc107"
                    elif score == 2: c = "#fd7e14"
                    else: c = "#dc3545"
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <p><strong>{question}</strong></p>
                        <div style="background-color: #e9ecef; border-radius: 5px; padding: 2px;">
                            <div style="width: {(score+1)*25}%; background-color: {c}; padding: 5px; border-radius: 3px; text-align: center; color: white;">
                                Score: {score}/3
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error analyzing user: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# Right Column - Images and Additional Info
with right_col:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    
    # Display subreddit info if available
    if 'current_post' in st.session_state:
        post = st.session_state.current_post
        try:
            subreddit = reddit.subreddit(post['subreddit'])
            if hasattr(subreddit, 'icon_img') and subreddit.icon_img:
                st.image(subreddit.icon_img, width=150, caption=f"r/{post['subreddit']}")
            st.markdown(f"""
            **Subreddit**: r/{post['subreddit']}  
            **Author**: u/{post['author']}  
            [View Post]({post['url']})
            """)
        except:
            st.markdown("Unable to fetch subreddit image")

    # Display user info if available
    if 'current_user' in st.session_state:
        try:
            user = reddit.redditor(st.session_state.current_user)
            if hasattr(user, 'icon_img') and user.icon_img:
                st.image(user.icon_img, width=150, caption=f"u/{st.session_state.current_user}")
            st.markdown(f"**Username**: u/{st.session_state.current_user}")
        except:
            st.markdown("Unable to fetch user image")
    
    st.markdown("</div>", unsafe_allow_html=True)
