import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import joblib
import requests
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find
try:
    find('tokenizers/punkt')
    find('corpora/stopwords')
    find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
nltk.download('punkt_tab')
from datetime import datetime, timedelta
import gdown
import os

api_key = 'AIzaSyDuyqqkQhp23johPm2V9Y_3cUK6-mc3d10'
youtube = build('youtube', 'v3', developerKey=api_key)

#vectorizer = joblib.load(r'C:\Users\GORKEM\Documents\VscodeProjects\commonAPP\Model Training\tfidf_vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')

lemmatizer = WordNetLemmatizer()

file_id = "1iSkNMFXU5BXNNE9OyubvmSEsCFDRk-5w"
destination = "tfidf_vectorizer.pkl"

download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

if not os.path.exists(destination):
    gdown.download(download_url, destination, quiet=False)

vectorizer = joblib.load(destination)

def get_channel_videos(channel_id, max_videos=50):
    videos = []
    next_page_token = None
    base_video_url = "https://www.youtube.com/watch?v="
    six_months_ago = datetime.now() - timedelta(days=3*30) 

    while len(videos) < max_videos:
        request = youtube.search().list(
            part='snippet',
            channelId=channel_id,
            maxResults=min(10, max_videos - len(videos)),
            type='video',
            order='date',
            pageToken=next_page_token
        ).execute()

        video_ids = [item['id']['videoId'] for item in request.get('items', []) if 'videoId' in item['id']]
        if not video_ids:
            break

        video_details_request = youtube.videos().list(
            part="contentDetails,snippet",
            id=",".join(video_ids)
        ).execute()

        for item, video_details in zip(request['items'], video_details_request['items']):
            if 'videoId' in item['id']:
                video_id = item['id']['videoId']
                video_title = item['snippet']['title']
                video_url = f"{base_video_url}{video_id}"

                # Get the publish date
                publish_date_str = video_details['snippet']['publishedAt']
                publish_date = datetime.strptime(publish_date_str, "%Y-%m-%dT%H:%M:%SZ")

                # Filter videos based on the date (only include videos from the last 6 months)
                if publish_date < six_months_ago:
                    continue

                # Get video duration
                duration = video_details['contentDetails']['duration']

                # Filter out short videos (less than a minute)
                if 'M' not in duration or (duration.startswith('PT') and 'S' in duration and 'M' not in duration):
                    continue

                videos.append({'video_id': video_id, 'video_title': video_title, 'video_url': video_url})

        next_page_token = request.get('nextPageToken')

        if not next_page_token:
            break

    return videos

def get_video_details(video_id):
    request = youtube.videos().list(part='statistics', id=video_id).execute()
    details = request['items'][0]['statistics']
    return details

def get_comments(video_id):
  
    comments = []
    
    try:
        
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100
        ).execute()

        
        for item in request['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        
        while 'nextPageToken' in request:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=request['nextPageToken']  
            ).execute()

            
            for item in request['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
        
        return comments

    except HttpError as e:
        
        if e.resp.status == 403 and 'commentsDisabled' in str(e):
            
            return None  
        else:
            
            raise

def extract_channel_id(url):
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        match = re.search(r'\"externalId\":\"(UC[\w-]+)\"', html_content)
        if match:
            return match.group(1)
    return


def analyze_sentiment(comments):
    if isinstance(comments, list):
        # Transform comments to a TF-IDF feature array
        comment_array = vectorizer.transform(comments)
    else:
        comment_array = vectorizer.transform([comments])

    # Use the model (classifier) to predict sentiment
    sentiment_scores = model.predict(comment_array)

    sentiment_mapping = {
        'positive': 1,
        'negative': 0,
    }

    # Convert sentiment predictions to numeric scores
    numeric_sentiment_scores = np.array([sentiment_mapping.get(score, 0) for score in sentiment_scores], dtype=np.float64)

    return np.mean(numeric_sentiment_scores) if len(numeric_sentiment_scores) > 0 else 0

#def is_video_sponsored(video_title, video_description, video_url):
   
    sponsored_keywords = [' is sponsored', 'paid promotion', 'partnered with', 'includes paid promotion', 'brand deal', 'paid partnership']
    
    combined_text = f"{video_title} {video_description}".lower()
    
    is_sponsored_by_keywords = any(keyword in combined_text for keyword in sponsored_keywords)

    is_sponsored_by_disclaimer = check_sponsorship_disclaimer(video_url)

    return is_sponsored_by_keywords or is_sponsored_by_disclaimer
    

def check_paid_promotion(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        # Send a GET request to the video page
        response = requests.get(url)
        
        # If the response status code is not 200, raise an HTTPError
        response.raise_for_status()
        
        # Search for 'paidContentOverlayRenderer' in the HTML content
        if re.search(r'paidContentOverlayRenderer', response.text):
            return True
        else:
            return False

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    
    return False

def get_video_description(video_id):
  
    
    request = youtube.videos().list(
        part='snippet',  
        id=video_id
    ).execute()

    
    if request['items']:
        return request['items'][0]['snippet']['description']
    else:
        return None
    
def get_video_engagement_metrics(video_id):
  
    try:
        request = youtube.videos().list(
            part="statistics",
            id=video_id
        ).execute()

        if 'items' in request and len(request['items']) > 0:
            stats = request['items'][0]['statistics']
            likes = int(stats.get('likeCount', 0))
            views = int(stats.get('viewCount', 0))
            comments_Count = int(stats.get('commentCount', 0))
            
            return likes, views, comments_Count
        else:
            return 0, 0, 0  
    except Exception as e:
        st.error(f"Error fetching engagement metrics for video {video_id}: {str(e)}")
        return 0, 0, 0



def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stopwords.words('english')]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)



def get_channel_name(channel_id):
  
    request = youtube.channels().list(part='snippet', id=channel_id).execute()
    if request['items']:
        return request['items'][0]['snippet']['title']
    else:
        return None


#----------------------------------------------------------------------------------------------------

def evaluate_channel(channel_id):
  
    videos = get_channel_videos(channel_id)  
    sponsored_sentiments = []
    unsponsored_sentiments = []
    sponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
    unsponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
    sponsored_videos = []
    unsponsored_videos = []

    for video in videos:
        video_id = video['video_id']
        video_title = video['video_title']
        video_url = video['video_url']
        video_description = get_video_description(video_id)
        

        
        #details = get_video_details(video_id)
        #comments = get_comments(video_id)
        
        #comments = comments.apply(clean_text)
        #comments = [clean_text(comment) for comment in comments]

        
        #sentiment_score = analyze_sentiment(comments)
        
        
        #if 'sponsored' in video_title.lower() or 'ad' in video_title.lower():
            #sponsored_sentiments.append(sentiment_score)
        #else:
            #unsponsored_sentiments.append(sentiment_score)

        comments = get_comments(video_id)    

        if comments is None:
            continue  

        comments = [clean_text(comment) for comment in comments]

        likes, views, comments_Count = get_video_engagement_metrics(video_id)   


        if check_paid_promotion(video_id):
            
            sponsored_sentiments.append(analyze_sentiment(comments))

            sponsored_videos.append(video)

            sponsored_engagement_metrics['likes'] += likes
            sponsored_engagement_metrics['views'] += views
            sponsored_engagement_metrics['comments'] += comments_Count
            sponsored_engagement_metrics['count'] += 1

        else:

            unsponsored_sentiments.append(analyze_sentiment(comments))

            unsponsored_videos.append(video)

            unsponsored_engagement_metrics['likes'] += likes
            unsponsored_engagement_metrics['views'] += views
            unsponsored_engagement_metrics['comments'] += comments_Count
            unsponsored_engagement_metrics['count'] += 1


    num_sponsored = len(sponsored_videos)
    num_unsponsored = len(unsponsored_videos)   

    #st.write(f"Total Sponsored Videos: {num_sponsored}")
    #st.write(f"Total Unsponsored Videos: {num_unsponsored}")
    #st.write(f"Total Sponsored Sentiment Videos: {len(sponsored_sentiments)}")
    #st.write(f"Total Unsponsored Sentiment Videos: {len(unsponsored_sentiments)}")
    
   
    avg_sponsored_sentiment = np.mean(sponsored_sentiments) if sponsored_sentiments else 0
    avg_unsponsored_sentiment = np.mean(unsponsored_sentiments) if unsponsored_sentiments else 0

    if sponsored_engagement_metrics['views'] > 0:
        avg_sponsored_engagement_score = (sponsored_engagement_metrics['likes'] + sponsored_engagement_metrics['comments']) / sponsored_engagement_metrics['views']
    else:
        avg_sponsored_engagement_score = 0

    
    if unsponsored_engagement_metrics['views'] > 0:
        avg_unsponsored_engagement_score = (unsponsored_engagement_metrics['likes'] + unsponsored_engagement_metrics['comments']) / unsponsored_engagement_metrics['views']
    else:
        avg_unsponsored_engagement_score = 0


    #avg_sponsored_engagement_score = np.mean(sponsored_engagement_metrics) if sponsored_engagement_metrics else 0
    #avg_unsponsored_engagement_score = np.mean(unsponsored_engagement_metrics) if unsponsored_engagement_metrics else 0
  

    #avg_sentiment_score = (avg_sponsored_sentiment + avg_unsponsored_sentiment) / 2

    overall_sponsored_score = (0.5 * avg_sponsored_sentiment) + (0.5 *  avg_sponsored_engagement_score)
    overall_unsponsored_score = (0.5 * avg_unsponsored_sentiment) + (0.5 *  avg_unsponsored_engagement_score)


    return overall_sponsored_score, overall_unsponsored_score, num_sponsored, num_unsponsored


st.title("YouTube Partner Estimator Tool")

st.subheader("Input Channel URL")
channel_url = st.text_input("Enter YouTube Channel URL")

if st.button("Evaluate"):

    channel_id = extract_channel_id(channel_url)
    channel_name = get_channel_name(channel_id)

    #avg_sponsored_sentiment, avg_unsponsored_sentiment = evaluate_channel(channel_id)
    overall_sponsored_score, overall_unsponsored_score, num_sponsored, num_unsponsored  = evaluate_channel(channel_id)

    #st.write(f"Average Sponsored Sentiment Score: {avg_sponsored_sentiment}")
    #st.write(f"Average Unsponsored Sentiment Score: {avg_unsponsored_sentiment}")

    st.subheader("Results")
    #st.write(f"YouTuber Name: {channel_name}")
    #st.write(f"Average Sponsored Content Score: {overall_sponsored_score}")
    #st.write(f"Average Unsponsored Content Score: {overall_unsponsored_score}")
    
    # After evaluating the channel, you can display the results in a table
    results = pd.DataFrame({
        'A': ['Overall Sponsored Score', 'Overall Unsponsored Score', 'YouTuber Name','Channel ID','Total Sponsored Videos','Total Unsponsored Videos'],
        'B': [overall_sponsored_score, overall_unsponsored_score, channel_name, channel_id, num_sponsored, num_unsponsored]

    })

    st.dataframe(results)


    if overall_sponsored_score > overall_unsponsored_score:
        st.success("This YouTuber is a potential good partner for sponsored videos!")
    else:
        st.warning("This YouTuber may not be a good fit for sponsored videos.")
