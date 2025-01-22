import re
import pandas as pd
import matplotlib.pyplot as plt
import logging
from googleapiclient.discovery import build
from transformers import pipeline
import textwrap

# === Setup Logging ===
logging.basicConfig(
    filename="app_logs.log",  # Log file name
    level=logging.INFO,  # Log info, warnings, and errors
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Replace with your API Key
API_KEY = "AIzaSyAlKTUhY9t3yaJvk0E2goCuLEtcsTOFMBM"

# Load Hugging Face Sentiment Model
try:
    sentiment_classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
        top_k=None
    )
    logging.info("Sentiment analysis model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load sentiment model: {e}")
    raise RuntimeError("Error loading sentiment model. Check logs for details.")

# Extract Video ID from URL

def extract_video_id(url):
    """
    Extracts YouTube video ID from various YouTube URL formats.
    """
    try:
        # Handle multiple YouTube URL formats
        patterns = [
            r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)",  
            r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?]+)",   
            r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^?]+)",       
            r"(?:https?:\/\/)?youtu\.be\/([^?]+)"                        
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                return video_id

        return None  # If no match found, return None
    except Exception as e:
        return None

# Fetch YouTube Comments with Pagination
def get_comments(video_id, max_results=500):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),  # Up to 100 per request
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        logging.info(f"Fetched {len(comments)} comments for Video ID: {video_id}")
    except Exception as e:
        logging.error(f"Error fetching comments: {e}")
        return [], f"Error fetching comments: {e}"
    
    return comments[:max_results], None


def get_video_title(video_id):
    """
    Fetches the title of the YouTube video using the YouTube Data API.
    """
    youtube = build("youtube", "v3", developerKey=API_KEY)

    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()

        if "items" in response and len(response["items"]) > 0:
            video_title = response["items"][0]["snippet"]["title"]
            return video_title
        else:
            return "Unknown Video Title"
    except Exception as e:
        logging.error(f"Error fetching video title: {e}")
        return "Error Fetching Title"

# Sentiment Analysis
def analyze_sentiment(comments):
    results = []
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    try:
        for comment in comments:
            sentiment_scores = sentiment_classifier(comment)[0]
            sentiment = max(sentiment_scores, key=lambda x: x['score'])
            sentiment_label = sentiment['label']
            sentiment_counts[sentiment_label] += 1
            results.append({"Comment": comment, "Sentiment": sentiment_label, "Score": sentiment['score']})

        logging.info("Sentiment analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return [], f"Error analyzing sentiment: {e}"

    return results, sentiment_counts

# Generate Pie Chart
def plot_pie_chart(sentiment_counts, video_title):
    """
    Generates a pie chart for sentiment distribution with a wrapped video title.
    """
    try:
        fig, ax = plt.subplots(figsize=(8,6))  # Increase figure size for better visibility

        # Wrap title if it's too long
        wrapped_title = "\n".join(textwrap.wrap(video_title, width=50))  # Wrap title every 50 characters

        ax.pie(
            sentiment_counts.values(), 
            labels=sentiment_counts.keys(), 
            autopct='%1.1f%%', 
            startangle=140
        )
        ax.set_title(f"Sentiment Analysis for:\n{wrapped_title}", fontsize=10)  # Apply wrapped title

        logging.info(f"Pie chart generated successfully for {video_title}.")
        return fig
    except Exception as e:
        logging.error(f"Error generating pie chart: {e}")
        return None

# Overall Sentiment Summary
def get_overall_sentiment(sentiment_counts):
    try:
        overall_sentiment = f"Overall Video Sentiment: {max(sentiment_counts, key=sentiment_counts.get).upper()}"
        logging.info(f"Overall Sentiment: {overall_sentiment}")
        return overall_sentiment
    except Exception as e:
        logging.error(f"Error calculating overall sentiment: {e}")
        return "Error calculating overall sentiment."