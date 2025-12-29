
import sys
import re
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
import matplotlib.pyplot as plt
from googleapiclient.discovery import build

sys.path.append(str(Path(__file__).parent))

from src.data_preprocessing import Vocabulary, clean_text
from src.model_bilstm_attention import BiLSTMAttention

API_KEY = "AIzaSyAlKTUhY9t3yaJvk0E2goCuLEtcsTOFMBM"
MODELS_DIR = Path(__file__).parent / "models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_trained_model():
    try:
        vocab_path = MODELS_DIR / "vocabulary.pkl"
        vocab = Vocabulary.load(vocab_path)
        
        model_path = MODELS_DIR / "BiLSTM_Attention_best.pt"
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        config = checkpoint['config'].copy()
        config.pop('bidirectional', None)

        model = BiLSTMAttention(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        return model, vocab
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

MODEL, VOCAB = load_trained_model()

def extract_video_id(url):
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([^?]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/v/([^?]+)",
        r"(?:https?://)?youtu\.be/([^?]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_title(video_id):
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if "items" in response and len(response["items"]) > 0:
            return response["items"][0]["snippet"]["title"]
        return "Unknown Video"
    except:
        return "Unknown Video"

def get_comments(video_id, max_results=100):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None
    try:
        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
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
        return comments[:max_results], None
    except Exception as e:
        return [], str(e)

def predict_sentiment(text, model, vocab, max_length=128):
    if model is None or vocab is None:
        return "neutral", [0.33, 0.34, 0.33]
    
    cleaned = clean_text(text)
    if not cleaned:
        return "neutral", [0.33, 0.34, 0.33]
    
    encoded = vocab.encode(cleaned)
    encoded = encoded + [0] * (max_length - len(encoded)) if len(encoded) < max_length else encoded[:max_length]
    
    input_ids = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    length = torch.tensor([min(len(vocab.encode(cleaned)), max_length)], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_ids, length)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return label_map[probabilities.argmax()], probabilities.tolist()

def analyze_comments(comments):
    results = []
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    
    for comment in comments:
        sentiment, probs = predict_sentiment(comment, MODEL, VOCAB)
        sentiment_counts[sentiment] += 1
        results.append({
            "Comment": comment[:100] + "..." if len(comment) > 100 else comment,
            "Sentiment": sentiment.capitalize(),
            "Confidence": f"{max(probs):.2%}"
        })
    return results, sentiment_counts

def plot_pie_chart(sentiment_counts, video_title):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
    chart_colors = [colors[label] for label in sentiment_counts.keys()]
    
    wedges, texts, autotexts = ax.pie(
        sentiment_counts.values(),
        labels=[label.capitalize() for label in sentiment_counts.keys()],
        autopct='%1.1f%%',
        startangle=140,
        colors=chart_colors,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    for autotext in autotexts:
        autotext.set_color('white')
    
    ax.set_title(f"Sentiment Analysis\n{video_title[:60]}", fontsize=14, fontweight='bold', pad=20)
    return fig

def get_overall_sentiment(sentiment_counts):
    total = sum(sentiment_counts.values())
    if total == 0: return "No comments analyzed"
    dominant = max(sentiment_counts, key=sentiment_counts.get)
    return f"Overall Sentiment: {dominant.upper()} ({sentiment_counts[dominant]/total*100:.1f}%)"

def youtube_sentiment_analysis(url, num_of_comments):
    if MODEL is None or VOCAB is None:
        return "Model not loaded.", None, None
    
    video_id = extract_video_id(url)
    if not video_id: return "Invalid URL", None, None
    
    video_title = get_video_title(video_id)
    comments, error = get_comments(video_id, int(num_of_comments))
    
    if error: return f"Error: {error}", None, None
    if not comments: return "No comments found", None, None
    
    results, sentiment_counts = analyze_comments(comments)
    chart = plot_pie_chart(sentiment_counts, video_title)
    summary = get_overall_sentiment(sentiment_counts)
    
    return summary, chart, pd.DataFrame(results).head(10)

example_urls = [
    ["https://www.youtube.com/watch?v=0e9WuB0Ua98"],
    ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"],
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center;'>YouTube Sentiment Analysis</h1>")
    gr.HTML("<p style='text-align: center;'>Custom Bi-LSTM + Attention Model | uabali </p>")
    
    with gr.Row():
        with gr.Column():
            youtube_url = gr.Textbox(label="YouTube Video URL")
            num_comments = gr.Slider(10, 500, step=10, value=100, label="Max Comments")
            submit_btn = gr.Button("Analyze", variant="primary")
            gr.Examples(example_urls, inputs=youtube_url)
        
        with gr.Column():
            output_summary = gr.Textbox(label="Summary")
            output_chart = gr.Plot(label="Chart")
            output_table = gr.Dataframe(label="Top 10 Comments")
    
    submit_btn.click(
        youtube_sentiment_analysis,
        inputs=[youtube_url, num_comments],
        outputs=[output_summary, output_chart, output_table]
    )

if __name__ == "__main__":
    demo.launch(share=False)