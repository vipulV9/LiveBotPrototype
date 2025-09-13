
import os
import time
import json
import numpy as np
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import google.generativeai as genai
from google.cloud import texttospeech
from google.api_core import exceptions as gcloud_exceptions
import redis
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ===================== LOAD ENV VARIABLES =====================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# ===================== CONFIGURE LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== CONFIGURE GEMINI =====================
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set in .env")
    raise ValueError("GEMINI_API_KEY env var not set")

genai.configure(api_key=GEMINI_API_KEY)

# ===================== CONFIGURE GOOGLE TTS =================
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    raise ValueError("Set GOOGLE_APPLICATION_CREDENTIALS env var")

# ===================== REDIS ================================
REDIS_HOST = "redis-16084.c52.us-east-1-4.ec2.redns.redis-cloud.com"
REDIS_PORT = 16084
REDIS_USERNAME = "default"

try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        db=0,
        decode_responses=True
    )
    r.ping()
    logger.info("Connected to Redis Cloud successfully")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Redis connection error: {e}")
    raise

# ===================== SENTENCE TRANSFORMERS =================
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Initialized SentenceTransformer model 'all-MiniLM-L6-v2'")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer: {e}")
    embedder = None

# ===================== FLASK APP ===========================
app = Flask(__name__)
AUDIO_FOLDER = "static/audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# HTML (Keep your existing HTML code here as-is)
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: #000;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 800px;
            text-align: center;
        }

        h1 {
            font-size: 2.5rem;
            margin-bottom: 2rem;
            color: #fff;
            font-weight: 300;
            letter-spacing: 1px;
        }

        .input-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 2rem;
        }

        .input-options {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            margin-bottom: 20px;
        }

        .mic-container {
            position: relative;
            width: 120px;
            height: 120px;
        }

        .mic-button {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: #1a1a1a;
            border: 2px solid #333;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }

        .mic-button:hover {
            background: #222;
            transform: scale(1.02);
        }

        .mic-button.recording {
            background: #1a1a1a;
            border-color: #555;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.2);
            }
            70% {
                box-shadow: 0 0 0 20px rgba(255, 255, 255, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(255, 255, 255, 0);
            }
        }

        .text-input-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .text-input {
            width: 300px;
            padding: 12px 15px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 1rem;
            margin-bottom: 10px;
        }

        .text-input:focus {
            outline: none;
            border-color: #4a90e2;
        }

        .send-button {
            padding: 10px 20px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .send-button:hover {
            background: #222;
            border-color: #555;
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .divider {
            width: 1px;
            height: 100px;
            background: #333;
        }

        .status {
            margin-top: 20px;
            font-size: 1.2rem;
            height: 30px;
            color: #aaa;
        }

        .response-container {
            background: #111;
            border: 1px solid #333;
            border-radius: 15px;
            padding: 25px;
            margin-top: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            min-height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .user-question, .ai-response {
            margin: 15px 0;
            padding: 15px;
            border-radius: 10px;
            text-align: left;
        }

        .user-question {
            background: #1a1a1a;
            border-left: 4px solid #4a90e2;
            align-self: flex-end;
            max-width: 80%;
            margin-left: auto;
        }

        .ai-response {
            background: #1a1a1a;
            border-left: 4px solid #5cb85c;
            align-self: flex-start;
            max-width: 80%;
        }

        .audio-player {
            margin-top: 20px;
            width: 100%;
        }

        .audio-player audio {
            width: 100%;
            border-radius: 10px;
            background: #1a1a1a;
        }

        .error {
            color: #ff6b6b;
            margin-top: 15px;
            padding: 10px;
            background: #1a1a1a;
            border: 1px solid #ff6b6b;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Assistant</h1>

        <div class="input-section">
            <div class="input-options">
                <div class="mic-container">
                    <div id="mic-button" class="mic-button">🎤</div>
                </div>

                <div class="divider"></div>

                <div class="text-input-container">
                    <input type="text" id="text-input" class="text-input" placeholder="Type your message">
                    <button id="send-button" class="send-button">Send</button>
                </div>
            </div>
        </div>

        <div id="status" class="status">Choose voice or text input</div>

        <div id="response-container" class="response-container" style="display: none;">
            <div id="user-question" class="user-question"></div>
            <div id="ai-response" class="ai-response"></div>
            <div id="audio-player" class="audio-player"></div>
        </div>

        <div id="error-container" class="error" style="display: none;"></div>
    </div>

    <script>
        const micButton = document.getElementById('mic-button');
        const textInput = document.getElementById('text-input');
        const sendButton = document.getElementById('send-button');
        const statusDiv = document.getElementById('status');
        const responseContainer = document.getElementById('response-container');
        const userQuestionDiv = document.getElementById('user-question');
        const aiResponseDiv = document.getElementById('ai-response');
        const audioPlayerDiv = document.getElementById('audio-player');
        const errorContainer = document.getElementById('error-container');

        let recognizing = false;
        let recognition;

        // Initialize speech recognition
        if ('webkitSpeechRecognition' in window) {
            recognition = new webkitSpeechRecognition();
        } else if ('SpeechRecognition' in window) {
            recognition = new SpeechRecognition();
        }

        if (recognition) {
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onstart = function() {
                recognizing = true;
                micButton.classList.add('recording');
                statusDiv.textContent = 'Listening... Speak now';
                errorContainer.style.display = 'none';
            };

            recognition.onend = function() {
                recognizing = false;
                micButton.classList.remove('recording');
                statusDiv.textContent = 'Processing...';
            };

            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                processUserInput(transcript);
            };

            recognition.onerror = function(event) {
                recognizing = false;
                micButton.classList.remove('recording');
                showError(`Microphone error: ${event.error}`);
            };
        } else {
            showError('Browser does not support SpeechRecognition API.');
        }

        // Microphone button click
        micButton.onclick = function() {
            if (recognizing) {
                recognition.stop();
            } else {
                responseContainer.style.display = 'none';
                recognition.start();
            }
        };

        // Send button click
        sendButton.onclick = function() {
            const text = textInput.value.trim();
            if (text) {
                processUserInput(text);
                textInput.value = '';
            }
        };

        // Enter key in text input
        textInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                sendButton.click();
            }
        });

        function processUserInput(input) {
            userQuestionDiv.textContent = `You: ${input}`;
            statusDiv.textContent = 'Getting response...';

            // Send to server for processing
            fetch("/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({"prompt": input})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showError(data.error);
                } else {
                    aiResponseDiv.textContent = `Assistant: ${data.response}`;

                    if (data.audio) {
                        audioPlayerDiv.innerHTML = `
                            <audio controls autoplay>
                                <source src="${data.audio}?t=${new Date().getTime()}" type="audio/mpeg">
                                Your browser does not support the audio element.
                            </audio>
                        `;
                    }

                    responseContainer.style.display = 'flex';
                    statusDiv.textContent = 'Choose voice or text input';
                }
            })
            .catch(error => {
                showError(`Error: ${error}`);
            });
        }

        function showError(message) {
            statusDiv.textContent = 'Error occurred';
            errorContainer.textContent = message;
            errorContainer.style.display = 'block';
            responseContainer.style.display = 'none';
        }
    </script>
</body>
</html>
'''

# ===================== EMBEDDINGS & COSINE SIM ===================
def embed(text):
    try:
        if embedder is None:
            raise ValueError("SentenceTransformer not initialized")
        embedding = embedder.encode(text, convert_to_numpy=True)
        logger.debug(f"Generated embedding for text: {text}")
        return embedding
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        return np.array([ord(c) % 50 for c in text[:100]], dtype=float)

def cosine_similarity(vec1, vec2):
    max_len = max(len(vec1), len(vec2))
    vec1 = np.pad(vec1, (0, max_len - len(vec1)), mode='constant')
    vec2 = np.pad(vec2, (0, max_len - len(vec2)), mode='constant')

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        logger.warning("Zero vector in cosine similarity")
        return 0.0

    score = np.dot(vec1, vec2) / (norm1 * norm2)
    logger.debug(f"Cosine similarity score: {score}")
    return score

# ===================== REDIS HELPERS ===========================
def redis_search(query, threshold=0.8):
    query_vec = embed(query)
    best_match = None
    best_score = 0

    try:
        for key in r.scan_iter("qa:*"):
            data = json.loads(r.get(key))
            vec = np.array(data["embedding"], dtype=float)
            score = cosine_similarity(query_vec, vec)
            logger.debug(f"Similarity for key {key}: {score}")
            if score > best_score:
                best_score = score
                best_match = data
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis search error: {e}")
        return None

    if best_match and best_score >= threshold:
        logger.info(f"Cache hit for query '{query}' with score {best_score}")
        return best_match
    logger.info(f"Cache miss for query '{query}'")
    return None

def redis_store(query, response, audio_path):
    key = f"qa:{hash(query)}"
    data = {
        "query": query,
        "response": response,
        "audio": audio_path,
        "embedding": embed(query).tolist()
    }
    try:
        r.setex(key, 24 * 3600, json.dumps(data))
        logger.info(f"Stored query '{query}' in Redis with TTL 24h")
    except redis.exceptions.RedisError as e:
        logger.error(f"Failed to store in Redis: {e}")

# ===================== AUDIO CLEANUP ===========================
def cleanup_audio(max_age_hours=24):
    import glob
    now = time.time()
    for file in glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3")):
        if os.path.getmtime(file) < now - (max_age_hours * 3600):
            os.remove(file)
            logger.info(f"Deleted old audio file: {file}")

# ===================== GEMINI RESPONSE =========================
def gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        response = model.generate_content(
            f"Answer only in one short sentence. No explanation. Question: {prompt}"
        )
        logger.info(f"Gemini generated response for prompt '{prompt}'")
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Gemini API error: {str(e)}"

# ===================== GOOGLE TTS =============================
def synthesize_speech(text, filename=None):
    try:
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-D"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config
        )
        filename = filename or f"reply_{int(time.time() * 1000)}.mp3"
        filepath = os.path.join(AUDIO_FOLDER, filename)
        with open(filepath, "wb") as out:
            out.write(response.audio_content)
        logger.info(f"Generated audio at {filepath}")
        return f"/static/audio/{filename}"
    except gcloud_exceptions.GoogleCloudError as e:
        logger.error(f"Google TTS error: {str(e)}")
        return f"Error with Google TTS: {str(e)}"

# ===================== FLASK ROUTES ============================
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/favicon.ico")
def favicon():
    favicon_path = os.path.join(app.root_path, "static", "favicon.ico")
    if os.path.exists(favicon_path):
        return send_from_directory("static", "favicon.ico")
    return "", 204

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        cached = redis_search(prompt)
        if cached:
            return jsonify({
                "response": cached["response"],
                "audio": cached["audio"]
            })

        response = gemini_response(prompt)
        if response.startswith("Gemini API error"):
            return jsonify({"error": response}), 500

        audio_path = synthesize_speech(response)
        if audio_path.startswith("Error"):
            return jsonify({"response": response, "audio": None}), 200

        redis_store(prompt, response, audio_path)
        cleanup_audio()

        return jsonify({"response": response, "audio": audio_path})
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ===================== RUN APP ================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
