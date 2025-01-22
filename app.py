import gradio as gr
import pandas as pd
import logging
from multilingual_sentiment_model import *

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Gradio Function with Logging
def youtube_sentiment_analysis(url, num_of_comments):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            logging.warning("Invalid YouTube URL entered in UI.")
            return "Error: Invalid YouTube URL", None, None

        video_title = get_video_title(video_id)  # Fetch video title

        comments, error = get_comments(video_id, int(num_of_comments))
        if error:
            logging.error(f"Error fetching comments: {error}")
            return f"Error fetching comments: {error}", None, None

        if not comments:
            logging.warning("No comments found for the video.")
            return "Error: No comments found.", None, None

        sentiment_results, sentiment_counts = analyze_sentiment(comments)
        chart = plot_pie_chart(sentiment_counts, video_title)  # Pass title to the chart
        summary = get_overall_sentiment(sentiment_counts)

        return summary, chart, pd.DataFrame(sentiment_results).head(5)

    except Exception as e:
        logging.exception(f"Unexpected Error: {str(e)}")
        return f"Unexpected Error: {str(e)}", None, None

# Gradio Interface (All Outputs Below Input)
iface = gr.Blocks()

# Example YouTube URLs 
example_urls = [
    "https://www.youtube.com/watch?v=0e9WuB0Ua98",
    "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=9bZkp7q19f0",
    "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
]

with iface:
    gr.Markdown("## YouTube Comment Sentiment Analysis", elem_classes='centered-title')

    gr.Markdown("Enter a YouTube video URL and specify the number of comments to analyze.")

    with gr.Row():
        youtube_url = gr.Textbox(label="YouTube Video URL")
        num_comments = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label="Number of Comments to Fetch")

    submit_btn = gr.Button("Submit")

    # All outputs are placed BELOW the input
    output_summary = gr.Textbox(label="Overall Sentiment Summary")
    output_chart = gr.Plot(label="Sentiment Chart")
    output_table = gr.Dataframe(label="Comment Sentiment Analysis")

    submit_btn.click(
        youtube_sentiment_analysis,
        inputs=[youtube_url, num_comments],
        outputs=[output_summary, output_chart, output_table],
    )

    gr.Markdown("### Example YouTube Video URLs for Testing (Click to Use)")
    with gr.Row():
        for example in example_urls:
            gr.Button(example).click(fn=lambda x=example: x, outputs=[youtube_url])

# Launch App
iface.launch(share=True)