import os
import cv2
import numpy as np
import tensorflow as tf
import base64
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# ---------------------------
# 1. Load the pre-trained model
# ---------------------------
model = tf.keras.models.load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ---------------------------
# 2. Create a Flask application and configure CORS
# ---------------------------
application = Flask(__name__)
CORS(application)

# Global variable to store prediction history
max_history = 10
# Initialize with non‑empty values so the charts have something to display
prediction_history = {label: [0] * max_history for label in emotion_labels}

# ---------------------------
# 3. Define the /predict API endpoint
# ---------------------------
@application.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if "image" not in data:
            logging.error("No image provided in the request")
            return jsonify({"error": "No image provided"}), 400

        # Decode the base64 image
        image_data = base64.b64decode(data["image"])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            logging.error("Failed to decode image data")
            return jsonify({"error": "Invalid image data"}), 400

        # Preprocess the image: resize to 48x48 and normalize
        resized = cv2.resize(frame, (48, 48)) / 255.0
        face = np.expand_dims(resized, axis=(0, -1))
        prediction = model.predict(face)
        predicted_index = np.argmax(prediction)
        emotion = emotion_labels[predicted_index]
        logging.debug(f"Predicted emotion: {emotion}")

        # Update prediction history: append a 1 for the predicted emotion and 0 for others
        for label in emotion_labels:
            value = 1 if label == emotion else 0
            prediction_history[label].append(value)
            if len(prediction_history[label]) > max_history:
                prediction_history[label].pop(0)

        return jsonify({"emotion": emotion})
    except Exception as e:
        logging.exception("Error processing /predict")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# 4. Create a Dash app using the Flask application
# ---------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__,
           server=application,
           url_base_pathname='/dashboard/',
           external_stylesheets=external_stylesheets)

# Define a darker blue color palette for chart elements.
dark_blue_palette = [
    '#0d3b66',  # Dark Blue (Start)
    '#0f4a75',
    '#125583',
    '#16688b',
    '#1a7fa0',
    '#1c84a8',
    '#1e6f9f'
]

# Set the chart background to white.
chart_bgcolor = '#ffffff'

# ---------------------------
# 5. Define the Dash layout (with updated chart positions and navbar)
# ---------------------------
app.layout = dbc.Container([
    # Navbar with logo and title in a flex container
    dbc.Navbar(
        dbc.Container([
            html.Div([
                html.Img(src="./assets/logo.png", height="40px", style={"marginRight": "15px"}),
                html.Span("Emotion Insights", style={"fontSize": "24px", "color": "white", "lineHeight": "40px"})
            ], style={"display": "flex", "alignItems": "center"})
        ], fluid=True),
        color="#343a40",
        dark=True,
        className="mb-4"
    ),

    # Row 1: Video Feed (left) and Current Emotion with Pie Chart (right)
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Live Video Feed", className="text-center mb-3"),
                html.Video(
                    id="video-feed",
                    autoPlay=True,
                    muted=True,  # Autoplay requires muted video in many browsers
                    style={"width": "100%", "borderRadius": "10px"}
                ),
            ], className="p-4 bg-white rounded shadow-sm"),
        ], width=6),
        dbc.Col([
            html.Div([
                html.H5("Current Emotion", className="text-center mb-3"),
                html.H2(id="current-emotion", className="text-center text-primary"),
                dcc.Graph(id='emotion-pie-chart', config={'displayModeBar': False})
            ], className="p-4 bg-white rounded shadow-sm"),
        ], width=6),
    ], className="mb-4"),

    # Row 2: Line Chart (left) and Heatmap (right)
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id='emotion-line-chart', config={'displayModeBar': False}),
            ], className="p-4 bg-white rounded shadow-sm"),
        ], width=6),
        dbc.Col([
            html.Div([
                dcc.Graph(id='emotion-heatmap', config={'displayModeBar': False}),
            ], className="p-4 bg-white rounded shadow-sm"),
        ], width=6),
    ], className="mb-4"),

    # Hidden div for the clientside callback output
    html.Div(id="dummy-output", style={"display": "none"}),

    # Interval component to trigger updates every second
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0)
], fluid=True)

# ---------------------------
# 6. Clientside callback: Capture video frame and send to /predict
# ---------------------------
app.clientside_callback(
    """
    function(n_intervals) {
        var video = document.getElementById('video-feed');
        if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
            return "";
        }
        // Create a canvas to capture the video frame
        var canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        var base64Image = canvas.toDataURL("image/jpeg").split(",")[1];

        // Send the captured frame to the /predict endpoint
        fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ image: base64Image })
        })
        .then(response => response.json())
        .then(data => {
            if (data.emotion) {
                document.getElementById("current-emotion").innerText = data.emotion;
            } else if (data.error) {
                console.error("Prediction error:", data.error);
            }
        })
        .catch(error => console.error("Error in fetch:", error));
        return "";
    }
    """,
    Output("dummy-output", "children"),
    Input("interval-component", "n_intervals")
)

# ---------------------------
# 7. application-side callback: Update the graphs
# ---------------------------
@app.callback(
    [Output('emotion-line-chart', 'figure'),
     Output('emotion-pie-chart', 'figure'),
     Output('emotion-heatmap', 'figure')],
    Input("interval-component", "n_intervals")
)
def update_graphs(n_intervals):
    # X-axis values for the history window
    x_range = list(range(max_history))

    # Build the line chart traces (one for each emotion) with dark blue tones
    line_traces = []
    for i, label in enumerate(emotion_labels):
        y_data = prediction_history[label]
        line_traces.append(go.Scatter(
            x=x_range,
            y=y_data,
            mode='lines+markers',
            name=label,
            line=dict(color=dark_blue_palette[i]),
            marker=dict(color=dark_blue_palette[i])
        ))
    line_fig = {
        'data': line_traces,
        'layout': go.Layout(
            title="Emotion Trends Over Time",
            xaxis={'title': 'Time Intervals'},
            yaxis={'title': 'Frequency'},
            margin=dict(l=40, r=20, t=40, b=30),
            paper_bgcolor=chart_bgcolor,
            plot_bgcolor=chart_bgcolor
        )
    }

    # Build the pie chart (donut style) with dark blue palette colors
    latest_counts = [sum(prediction_history[label]) for label in emotion_labels]
    if sum(latest_counts) == 0:
        latest_counts = [1] * len(emotion_labels)
    pie_fig = {
        'data': [go.Pie(
            labels=emotion_labels,
            values=latest_counts,
            hole=0.4,
            marker=dict(colors=dark_blue_palette)
        )],
        'layout': go.Layout(
            title="Emotion Distribution",
            margin=dict(l=40, r=20, t=40, b=30),
            paper_bgcolor=chart_bgcolor,
            plot_bgcolor=chart_bgcolor
        )
    }

    # Build the heatmap: each row represents an emotion’s history (using the 'Blues' colorscale)
    heatmap_data = [prediction_history[label] for label in emotion_labels]
    heatmap_fig = {
        'data': [go.Heatmap(
            z=heatmap_data,
            x=x_range,
            y=emotion_labels,
            colorscale='Blues'
        )],
        'layout': go.Layout(
            title="Emotion Frequency Heatmap",
            xaxis={'title': 'Time Intervals'},
            margin=dict(l=40, r=20, t=40, b=30),
            paper_bgcolor=chart_bgcolor,
            plot_bgcolor=chart_bgcolor
        )
    }

    return line_fig, pie_fig, heatmap_fig

@application.route("/")
def home():
    return render_template("index.html")

# ---------------------------
# 8. Run the app
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", port=port, debug=True)