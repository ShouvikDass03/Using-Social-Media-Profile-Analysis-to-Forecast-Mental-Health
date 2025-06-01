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
import torch
from torch import nn

# Page Configuration
st.set_page_config(
    page_title="Social Media Profile for Predicting the Psychological State",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Downloads and Setup
nltk.download('stopwords')
nltk.download('punkt_tab')

# Helper function to display post and result consistently
def display_post_and_result(post_data):
    st.markdown(f"<div class='post-title'>📝 {post_data['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='post-content'>{post_data['content']}</div>", unsafe_allow_html=True)
    
    if 'risk_result' in post_data:
        result = post_data['risk_result']
        result_color = "red" if result == 1 else "green"
        result_text = "🚨 High Risk Indicators Detected" if result == 1 else "✅ No Significant Risk Detected"
        
        st.markdown(f"""
        <div class="result-box" style="background-color: {'#ffe6e6' if result == 1 else '#e6ffe6'}">
            <h3 style="color: {result_color}; text-align: center;">{result_text}</h3>
        </div>
        """, unsafe_allow_html=True)

# Secrets and Device Setup
device = "cpu"  # We'll stick with CPU for now
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

# Initialize the SentenceTransformer model
@st.cache_resource
def load_sentence_transformer(model_name):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

embedder = load_sentence_transformer('all-MiniLM-L6-v2')
if embedder is None:
    st.error("Failed to initialize the sentence transformer model. Some functionality may be limited.")

# BDI-II Questions List
bdi_questions = [
    ("Sadness", [
        "I do not feel sad",
        "I feel sad",
        "I am sad all the time",
        "I am so sad or unhappy that I can't stand it"
    ]),
    ("Pessimism", [
        "I am not discouraged about my future",
        "I feel more discouraged about my future than I used to",
        "I do not expect things to work out for me",
        "I feel my future is hopeless"
    ]),
    ("Past failure", [
        "I do not feel like a failure",
        "I have failed more than I should have",
        "As I look back, I see a lot of failures",
        "I feel I am a complete failure"
    ]),
    ("Loss of pleasure", [
        "I get as much pleasure as I ever did from the things I enjoy",
        "I don't enjoy things as much as I used to",
        "I get very little pleasure from the things I used to enjoy",
        "I can't get any pleasure from the things I used to enjoy"
    ]),
    ("Guilty feelings", [
        "I don't feel particularly guilty",
        "I feel guilty over many things I have done or should have done",
        "I feel quite guilty most of the time",
        "I feel guilty all of the time"
    ]),
    ("Punishment feelings", [
        "I don't feel I am being punished",
        "I feel I may be punished",
        "I expect to be punished",
        "I feel I am being punished"
    ]),
    ("Self-dislike", [
        "I feel the same about myself as ever",
        "I have lost confidence in myself",
        "I am disappointed in myself",
        "I dislike myself"
    ]),
    ("Self-criticalness", [
        "I don't criticize or blame myself more than usual",
        "I am more critical of myself than I used to be",
        "I criticize myself for all of my faults",
        "I blame myself for everything bad that happens"
    ]),
    ("Suicidal thoughts", [
        "I don't have any thoughts of killing myself",
        "I have thoughts of killing myself, but I would not carry them out",
        "I would like to kill myself",
        "I would kill myself if I had the chance"
    ]),
    ("Crying", [
        "I don't cry anymore than I used to",
        "I cry more than I used to",
        "I cry over every little thing",
        "I feel like crying but I can't"
    ]),
    ("Agitation", [
        "I am no more restless or wound up than usual",
        "I feel more restless or wound up than usual",
        "I am so restless or agitated that it's hard to stay still",
        "I am so restless or agitated that I have to keep moving or doing something"
    ]),
    ("Loss of interest", [
        "I have not lost interest in other people or activities",
        "I am less interested in other people or things than before",
        "I have lost most of my interest in other people or things",
        "It's hard to get interested in anything"
    ]),
    ("Indecisiveness", [
        "I make decisions about as well as ever",
        "I find it more difficult to make decisions than usual",
        "I have much greater difficulty in making decisions",
        "I can't make decisions at all anymore"
    ]),
    ("Worthlessness", [
        "I do not feel I am worthless",
        "I don't consider myself as worthwhile and useful as I used to",
        "I feel more worthless as compared to others",
        "I feel utterly worthless"
    ]),
    ("Loss of energy", [
        "I have as much energy as ever",
        "I have less energy than I used to have",
        "I don't have enough energy to do very much",
        "I don't have enough energy to do anything"
    ]),
    ("Changes in sleeping pattern", [
        "I have not experienced any change in my sleeping pattern",
        "I sleep a little more or less than usual",
        "I sleep a lot more or less than usual",
        "I sleep most of the day or wake up early and can't get back to sleep"
    ]),
    ("Irritability", [
        "I am no more irritable than usual",
        "I am more irritable than usual",
        "I am much more irritable than usual",
        "I am irritable all the time"
    ]),
    ("Changes in appetite", [
        "I have not experienced any change in my appetite",
        "My appetite is somewhat less or greater than usual",
        "My appetite is much less or greater than before",
        "I have no appetite at all or I crave food all the time"
    ]),
    ("Concentration difficulty", [
        "I can concentrate as well as ever",
        "I can't concentrate as well as usual",
        "It's hard to keep my mind on anything for long",
        "I find I can't concentrate on anything"
    ]),
    ("Tiredness or fatigue", [
        "I am no more tired or fatigued than usual",
        "I get tired or fatigued more easily than usual",
        "I am too tired or fatigued to do a lot of the things I used to do",
        "I am too tired or fatigued to do most of the things I used to do"
    ]),
    ("Loss of interest in sex", [
        "I have not noticed any recent change in my interest in sex",
        "I am less interested in sex than I used to be",
        "I am much less interested in sex now",
        "I have lost interest in sex completely"
    ])
]

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
            margin: 0;
        }
        .block {
            background-color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            font-size: 2.5rem;
            margin: 1.5rem 0;
            font-weight: bold;
        }
        .subtitle {
            color: #34495e;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 3rem;
        }
        .section-title {
            color: #2c3e50;
            font-size: 1.5rem;
            margin: 1.5rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #eee;
        }
        /* Professional button styling */
        .stButton > button {
            width: auto;
            min-width: 200px;
            padding: 0.5rem 2rem;
            font-size: 1rem;
            font-weight: 500;
            color: white;
            background: linear-gradient(135deg, #FF4B4B 0%, #FF3333 100%);
            border: none;
            border-radius: 25px;
            box-shadow: 0 2px 5px rgba(255, 75, 75, 0.2);
            transition: all 0.3s ease;
            margin: 1rem 0;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
            background: linear-gradient(135deg, #FF3333 0%, #FF2929 100%);
        }
        .stButton > button:active {
            transform: translateY(0);
            box-shadow: 0 2px 5px rgba(255, 75, 75, 0.2);
        }
        /* Input field styling */
        .stTextInput > div > div > input {
            padding: 0.75rem 1rem;
            font-size: 1rem;
            border-radius: 8px;
            border: 2px solid #eee;
            transition: all 0.3s ease;
        }
        .stTextInput > div > div > input:focus {
            border-color: #FF4B4B;
            box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.1);
        }
        /* Container spacing */
        div[data-testid="stVerticalBlock"] > div {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        /* Warning box styling */
        .warning {
            background-color: #fff3cd;
            color: #856404;
            padding: 1.25rem;
            border-radius: 8px;
            border-left: 5px solid #ffc107;
            margin: 1.5rem 0;
        }
        /* Post display styling */
        .post-box {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            border: 1px solid #e9ecef;
        }
        .post-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #1e1e1e;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid #e9ecef;
        }
        .post-content {
            color: #444;
            line-height: 1.6;
            white-space: pre-wrap;
            font-size: 1.1em;
            padding: 0.5rem 0;
        }
        /* Result box styling */
        .result-box {
            padding: 1.25rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        /* Column styling */
        div[data-testid="column"] {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.75rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Main Layout
st.markdown('<h1 class="title">Social Media Profile for Predicting the Psychological State</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyzing Social Media Content for Mental Health Indicators</p>', unsafe_allow_html=True)

# Create three columns with proper spacing
left_col, middle_col, right_col = st.columns([1, 2, 1], gap="large")

# Left Column - Project Description and Disclaimers
with left_col:
    st.markdown('<div class="section-title">About the Project</div>', unsafe_allow_html=True)
    st.markdown("""
    This system uses advanced machine learning algorithms to analyze social media content for potential mental health indicators. It provides two main functionalities:
    
    1. **Subreddit Post Analysis**: Analyzes random posts from specified subreddits for potential suicide risk indicators.
    
    2. **User BDI-II Assessment**: Estimates a user's BDI-II (Beck Depression Inventory-II) score based on their Reddit post history.
    """)
    
    st.markdown('<div class="section-title">Important Disclaimers</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warning">
    ⚠️ This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
    
    🏥 If you or someone you know is experiencing a mental health crisis:
    - Call 9152987821 (National Suicide & Crisis Lifeline)
    - Seek immediate professional help
    </div>
    """, unsafe_allow_html=True)

# Middle Column - Search Functionalities
with middle_col:
    # Subreddit Analysis Section
    st.markdown('<div class="section-title">🔍 Subreddit Post Analysis</div>', unsafe_allow_html=True)
    subreddit_name = st.text_input("Enter subreddit name", value='AskReddit', key='subreddit_input').strip().replace(" ", "")
    analyze_button = st.button('Fetch and predict')

    if analyze_button:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            _ = subreddit.id
            random_post = None
            try:
                # Collect posts from different categories
                all_posts = []
                
                # Get hot posts
                try:
                    hot_posts = list(subreddit.hot(limit=20))
                    all_posts.extend(hot_posts)
                except:
                    st.warning("Could not fetch hot posts")
                
                # Get new posts
                try:
                    new_posts = list(subreddit.new(limit=20))
                    all_posts.extend(new_posts)
                except:
                    st.warning("Could not fetch new posts")
                
                # Get top posts of all time
                try:
                    top_posts = list(subreddit.top(time_filter='all', limit=20))
                    all_posts.extend(top_posts)
                except:
                    st.warning("Could not fetch top posts")
                
                if all_posts:
                    random_post = random.choice(all_posts)
                else:
                    st.error(f"No posts found in r/{subreddit_name}")
                    st.stop()
            except Exception as e:
                st.error(f"Error: {e}")

            if random_post is None:
                st.error(f"No posts found in r/{subreddit_name}")
                st.stop()

            post_content = random_post.title + " " + random_post.selftext
            
            # Store post details in session state and clear previous user data
            if 'posts_data' not in st.session_state:
                st.session_state.posts_data = {}
            
            # Clear previous user data when fetching new post
            st.session_state.posts_data.pop('current_user', None)
            
            # Analysis
            transformed_post = transform_text(post_content)
            vector_input = tfidf.transform([transformed_post])
            result = model.predict(vector_input.toarray())[0]
            
            st.session_state.posts_data['current_post'] = {
                'subreddit': str(random_post.subreddit),
                'author': str(random_post.author),
                'content': random_post.selftext,
                'title': random_post.title,
                'url': f"https://www.reddit.com{random_post.permalink}",
                'risk_result': result
            }

            # Display post content and result
            display_post_and_result(st.session_state.posts_data['current_post'])

        except Exception as e:
            st.error(f"Error: {e}")

    # Display existing post if available
    elif 'posts_data' in st.session_state and 'current_post' in st.session_state.posts_data:
        display_post_and_result(st.session_state.posts_data['current_post'])

    # BDI-II Analysis Section
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
            if 'posts_data' not in st.session_state:
                st.session_state.posts_data = {}
            st.session_state.posts_data['current_user'] = reddit_user

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

# Right Column - Images and Additional Info
with right_col:
    # Only show content if there's data to display
    if 'posts_data' in st.session_state and ('current_post' in st.session_state.posts_data or 'current_user' in st.session_state.posts_data):
        
        # Display subreddit info if available
        if 'current_post' in st.session_state.posts_data:
            post = st.session_state.posts_data['current_post']
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
        if 'current_user' in st.session_state.posts_data:
            try:
                user = reddit.redditor(st.session_state.posts_data['current_user'])
                if hasattr(user, 'icon_img') and user.icon_img:
                    st.image(user.icon_img, width=150, caption=f"u/{st.session_state.posts_data['current_user']}")
                st.markdown(f"**Username**: u/{st.session_state.posts_data['current_user']}")
            except:
                st.markdown("Unable to fetch user image")
        
