import streamlit as st
import praw
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer, util
import pickle
import random
import pandas as pd
import numpy as np
from PIL import Image
import time

# Downloads
nltk.download('stopwords')
nltk.download('punkt_tab')

# Secrets (stored in .streamlit/secrets.toml)
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

# Reddit setup
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

# Sentence Transformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# BDI-II Full List
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

# Page configuration
st.set_page_config(
    page_title="Suicide Risk Detection System",
    page_icon="🤍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        padding: 15px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        padding: 0.5rem 2rem;
        font-size: 16px;
        border-radius: 5px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #FF3333;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    h1 {
        color: #1E1E1E;
    }
    .alert {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-info {
        background-color: #E8F4F9;
        border-left: 5px solid #2196F3;
    }
    .alert-warning {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application uses advanced machine learning to detect potential suicide risk indicators in social media posts.
    
    ### How it works:
    1. Enter the text you want to analyze
    2. Click 'Analyze Text'
    3. View the detailed analysis results
    
    ### Important Note:
    This tool is for educational purposes only and should not be used as a substitute for professional medical advice.
    """)
    
    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("""
    If you or someone you know needs help, please contact:
    - National Suicide Prevention Lifeline (US): 988
    - Crisis Text Line: Text HOME to 741741
    """)

# Main content
st.title("🤍 Social Media Suicide Risk Detection")
st.markdown("""
<div class="alert alert-info">
This tool analyzes text content to identify potential indicators of suicide risk. 
Please use this tool responsibly and seek professional help if you or someone you know is in crisis.
</div>
""", unsafe_allow_html=True)

# Text input section
text_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Type or paste the text content here..."
)

col1, col2, col3 = st.columns([1,1,1])
with col2:
    analyze_button = st.button("Analyze Text")

# Analysis section
if analyze_button and text_input:
    with st.spinner('Analyzing text...'):
        # Add your model prediction logic here
        # For now, we'll use a placeholder
        time.sleep(2)  # Simulating processing time
        
        st.markdown("### Analysis Results")
        
        # Example results display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="alert alert-warning">
            <h4>Risk Assessment</h4>
            <p>Based on the analysis, this text shows potential indicators that require attention.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Example confidence score
            confidence = 0.75  # Replace with actual model confidence
            st.markdown("#### Confidence Score")
            st.progress(confidence)
            st.markdown(f"Confidence: {confidence*100:.1f}%")
        
        # Detailed analysis
        st.markdown("### Detailed Analysis")
        with st.expander("View Details"):
            st.markdown("""
            - **Sentiment Analysis**: Negative
            - **Key Indicators Detected**: 3
            - **Context Assessment**: Medium Risk
            """)
            
            # Example visualization
            data = pd.DataFrame({
                'Category': ['Negative', 'Neutral', 'Positive'],
                'Score': [0.7, 0.2, 0.1]
            })
            st.bar_chart(data.set_index('Category'))

elif analyze_button and not text_input:
    st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<small>Developed with ❤️ for mental health awareness. This is a prototype tool and should not be used as a substitute for professional medical advice.</small>
</div>
""", unsafe_allow_html=True)

# UI Section: BDI-II Mapping
st.title("🔍 BDI-II Estimate from Reddit User Posts")

reddit_user = st.text_input("Enter Reddit username for BDI-II mapping")

if st.button("Analyze User for BDI-II Estimate"):
    try:
        redditor = reddit.redditor(reddit_user)
        posts = [submission.title + " " + submission.selftext for submission in redditor.submissions.new(limit=25)]

        if not posts:
            st.error("No posts found for this user.")
            st.stop()

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

        # Display total score with progress bar
        st.subheader("🧠 Estimated BDI-II Score")

        # Determine severity and color
        if bdi_score <= 13:
            severity = "Minimal depression"
            color = "green"
        elif bdi_score <= 19:
            severity = "Mild depression"
            color = "yellow"
        elif bdi_score <= 28:
            severity = "Moderate depression"
            color = "orange"
        else:
            severity = "Severe depression"
            color = "red"

        # Display progress bar and label
        score_percent = int((bdi_score / 63) * 100)
        st.markdown(
            f"""
            <div style="background-color:lightgray; border-radius:8px; padding:4px 8px; margin-bottom:12px;">
                <div style="width:{score_percent}%; background-color:{color}; padding:6px; border-radius:4px; text-align:center; color:black;">
                    Total Score: {bdi_score} / 63 &nbsp;—&nbsp; <strong>{severity}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📊 Detailed BDI-II Item Breakdown"):
            for question, score in bdi_breakdown:
                if score == 0:
                    color = "green"
                elif score == 1:
                    color = "yellow"
                elif score == 2:
                    color = "orange"
                else:
                    color = "red"

                st.markdown(f"**{question}**")
                st.markdown(
                    f"""
                    <div style="background-color:lightgray; border-radius:8px; padding:4px 8px; margin-bottom:8px;">
                        <div style="width:{(score+1)*25}%; background-color:{color}; padding:4px; border-radius:4px; text-align:center; color:black;">
                            Score: {score} / 3
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"Error analyzing user: {e}")
