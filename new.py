import os
import time
import json
import glob
import numpy as np
import requests
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
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# ===================== CONFIGURE LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== VALIDATE ENV VARIABLES =================
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set in .env")
    raise ValueError("GEMINI_API_KEY env var not set")

google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not google_creds:
    logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    raise ValueError("Set GOOGLE_APPLICATION_CREDENTIALS env var")
try:
    json.loads(google_creds)
    logger.info(f"GOOGLE_APPLICATION_CREDENTIALS validated (length: {len(google_creds)} chars)")
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS: {str(e)}")
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS must be valid JSON string")

# ===================== CONFIGURE GEMINI =====================
genai.configure(api_key=GEMINI_API_KEY)

# ===================== REDIS ================================
try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=int(REDIS_PORT),
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
embedder = None
def load_embedder():
    global embedder
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Initialized SentenceTransformer model 'all-MiniLM-L6-v2'")
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        embedder = None

# ===================== FLASK APP ===========================
app = Flask(__name__)
AUDIO_FOLDER = "static/audio"
try:
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    os.chmod(AUDIO_FOLDER, 0o755)
    logger.info(f"Created audio folder: {AUDIO_FOLDER} with permissions 755")
except OSError as e:
    logger.error(f"Failed to create audio folder {AUDIO_FOLDER}: {str(e)}")
    raise

# HTML (unchanged from previous discussions)
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
            0% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.2); }
            70% { box-shadow: 0 0 0 20px rgba(255, 255, 255, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
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

        micButton.onclick = function() {
            if (recognizing) {
                recognition.stop();
            } else {
                responseContainer.style.display = 'none';
                recognition.start();
            }
        };

        sendButton.onclick = function() {
            const text = textInput.value.trim();
            if (text) {
                processUserInput(text);
                textInput.value = '';
            }
        };

        textInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                sendButton.click();
            }
        });

        function processUserInput(input) {
            userQuestionDiv.textContent = `You: ${input}`;
            statusDiv.textContent = 'Getting response...';
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
                    audioPlayerDiv.innerHTML = '';
                    if (data.audio) {
                        audioPlayerDiv.innerHTML = `
                            <audio controls autoplay>
                                <source src="${data.audio}" type="audio/mpeg">
                                Your browser does not support the audio element.
                            </audio>
                        `;
                    }
                    if (data.song_url) {
                        audioPlayerDiv.innerHTML = `
                            <audio controls autoplay>
                                <source src="${data.song_url}" type="audio/mpeg">
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

# ===================== JIOSAAVN API ===========================
def search_jiosaavn_song(query):
    try:
        url = "https://saavn.dev/api/search/songs"
        params = {"query": query, "limit": 1}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("success") and data.get("data", {}).get("results"):
            song = data["data"]["results"][0]
            song_name = song.get("name", "Unknown Song")
            song_url = next((item["url"] for item in song.get("downloadUrl", []) if item["quality"] == "320kbps"), None)
            if song_url:
                logger.info(f"Found song '{song_name}' with URL: {song_url}")
                return song_name, song_url
            else:
                logger.warning(f"No 320kbps URL found for song '{song_name}'")
                return song_name, None
        else:
            logger.warning(f"No songs found for query '{query}'")
            return None, None
    except requests.exceptions.RequestException as e:
        logger.error(f"JioSaavn API error: {str(e)}")
        return None, None

# ===================== EMBEDDINGS & COSINE SIM ===================
def embed(text):
    global embedder
    if embedder is None:
        load_embedder()
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
        audio_path = best_match.get("audio")
        if audio_path and not os.path.exists(audio_path.lstrip('/')):
            logger.warning(f"Cached audio file {audio_path} missing, ignoring cache")
            return None
        logger.info(f"Cache hit for query '{query}' with score {best_score}")
        return best_match
    logger.info(f"Cache miss for query '{query}'")
    return None

def redis_store(query, response, audio_path=None, song_url=None):
    key = f"qa:{hash(query)}"
    data = {
        "query": query,
        "response": response,
        "audio": audio_path,
        "song_url": song_url,
        "embedding": embed(query).tolist()
    }
    try:
        r.setex(key, 24 * 3600, json.dumps(data))
        logger.info(f"Stored query '{query}' in Redis with TTL 24h")
    except redis.exceptions.RedisError as e:
        logger.error(f"Failed to store in Redis: {e}")

# ===================== AUDIO CLEANUP ===========================
def cleanup_audio(max_age_hours=24):
    now = time.time()
    for file in glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3")):
        if os.path.getmtime(file) < now - (max_age_hours * 3600):
            try:
                os.remove(file)
                logger.info(f"Deleted old audio file: {file}")
            except Exception as e:
                logger.error(f"Failed to delete audio file {file}: {str(e)}")

# ===================== GEMINI RESPONSE =========================
def gemini_response(prompt):
    try:
        prompt_lower = prompt.lower()
        if any(keyword in prompt_lower for keyword in ["play", "song", "music"]):
            song_name, song_url = search_jiosaavn_song(prompt)
            if song_name and song_url:
                return f"Playing '{song_name}'", None, song_url
            elif song_name:
                return f"Found song '{song_name}' but no playable URL available", None, None
            else:
                return "Sorry, I couldn't find that song", None, None
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        response = model.generate_content(prompt)
        logger.info(f"Gemini generated response for prompt '{prompt}'")
        return response.text.strip().replace("\n", " "), None, None
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Gemini API error: {str(e)}", None, None

# ===================== GOOGLE TTS =============================
def synthesize_speech(text, filename=None):
    try:
        if not text or len(text.strip()) == 0:
            logger.error("Empty text provided to Google TTS")
            return None
        if len(text) > 5000:
            logger.warning(f"Text too long for Google TTS ({len(text)} chars); truncating")
            text = text[:5000]

        logger.info(f"Synthesizing speech for text (length: {len(text)}): {text[:50]}...")
        try:
            client = texttospeech.TextToSpeechClient()
            logger.info("Google TTS client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS client: {str(e)}", exc_info=True)
            return None

        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-D"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        logger.info("Sending request to Google TTS API")
        try:
            response = client.synthesize_speech(
                input=input_text,
                voice=voice,
                audio_config=audio_config
            )
            logger.info(f"Received TTS response (audio content length: {len(response.audio_content)} bytes)")
        except Exception as e:
            logger.error(f"Google TTS API call failed: {str(e)}", exc_info=True)
            return None

        if len(response.audio_content) == 0:
            logger.error("Google TTS returned empty audio content")
            return None

        filename = filename or f"reply_{int(time.time() * 1000)}.mp3"
        filepath = os.path.join(AUDIO_FOLDER, filename)
        logger.info(f"Writing audio to {filepath}")
        try:
            with open(filepath, "wb") as out:
                out.write(response.audio_content)
                out.flush()
                os.fsync(out.fileno())
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"Generated audio at {filepath} (size: {file_size} bytes)")
                return f"/static/audio/{filename}"
            else:
                logger.error(f"File existence check failed for {filepath}")
                return None
        except IOError as e:
            logger.error(f"File I/O error writing audio to {filepath}: {str(e)}")
            return None
    except gcloud_exceptions.InvalidArgument as e:
        logger.error(f"Invalid input for Google TTS: {str(e)}")
        return None
    except gcloud_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied for Google TTS (check GOOGLE_APPLICATION_CREDENTIALS): {str(e)}")
        return None
    except gcloud_exceptions.QuotaExceeded as e:
        logger.error(f"Google TTS quota exceeded: {str(e)}")
        return None
    except gcloud_exceptions.GoogleAPIError as e:
        logger.error(f"Google TTS API error: {str(e)}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error in Google TTS: {str(e)}", exc_info=True)
        return None

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
            logger.error("No prompt provided in request")
            return jsonify({"error": "No prompt provided"}), 400

        cached = redis_search(prompt)
        if cached:
            logger.info(f"Returning cached response for prompt: {prompt}")
            return jsonify({
                "response": cached["response"],
                "audio": cached["audio"],
                "song_url": cached.get("song_url")
            })

        response, _, song_url = gemini_response(prompt)
        if response.startswith("Gemini API error"):
            logger.error(f"Gemini response error: {response}")
            return jsonify({"error": response}), 500

        logger.info(f"Gemini response for '{prompt}' (length: {len(response)}): {response[:50]}...")

        audio_path = None
        if not song_url:
            audio_path = synthesize_speech(response)
            if audio_path is None:
                logger.warning("No audio generated for response")
                return jsonify({"response": response, "audio": None, "song_url": None}), 200

        redis_store(prompt, response, audio_path, song_url)
        cleanup_audio()

        logger.info(f"Returning new response for prompt: {prompt}")
        return jsonify({"response": response, "audio": audio_path, "song_url": song_url})
    except Exception as e:
        logger.error(f"Server error in /chat: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/test_tts", methods=["POST"])
def test_tts():
    try:
        data = request.json
        text = data.get("text", "Hello, this is a test.")
        audio_path = synthesize_speech(text)
        if audio_path:
            logger.info(f"Test TTS successful, audio at {audio_path}")
            return jsonify({"audio": audio_path})
        else:
            logger.error("Test TTS failed to generate audio")
            return jsonify({"error": "Failed to generate audio"}), 500
    except Exception as e:
        logger.error(f"Test TTS error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Test TTS error: {str(e)}"}), 500

# ===================== RUN APP ================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
