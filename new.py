import os
import time
import json
import glob
import numpy as np
import requests
import tempfile
import logging
import redis
from flask import Flask, render_template_string, request, jsonify, send_from_directory, redirect
import base64
import google.generativeai as genai
from google.cloud import texttospeech
from google.api_core import exceptions as gcloud_exceptions

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

# Initialize Flask app
app = Flask(__name__)

# ===================== VALIDATE ENV VARIABLES =================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set in .env")
    raise ValueError("GEMINI_API_KEY env var not set")

# ===================== GOOGLE CREDENTIALS FIX =================
google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not google_creds:
    logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    raise ValueError("Set GOOGLE_APPLICATION_CREDENTIALS env var")

try:
    creds_dict = json.loads(google_creds)  # Parse JSON string from Config Var
    required_fields = ["type", "project_id", "private_key", "client_email", "client_id", "auth_uri", "token_uri"]
    missing_fields = [field for field in required_fields if field not in creds_dict]
    if missing_fields:
        logger.error(f"GOOGLE_APPLICATION_CREDENTIALS missing required fields: {missing_fields}")
        raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS missing fields: {missing_fields}")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as temp:
        json.dump(creds_dict, temp)
        temp_path = temp.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
    logger.info(f"Google credentials validated (client_email: {creds_dict['client_email']}, project_id: {creds_dict['project_id']})")
except json.JSONDecodeError:
    logger.error("Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS")
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS must be valid JSON string")

# ===================== CONFIGURE GEMINI =====================
genai.configure(api_key=GEMINI_API_KEY)

# ===================== REDIS ================================
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )
    r.ping()
    logger.info("Connected to Redis")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise

# ===================== SENTENCE TRANSFORMERS =================
embedder = None  # Lazy-load to save memory
def load_embedder():
    global embedder
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded sentence transformer model")
    except Exception as e:
        logger.error(f"Failed to load sentence transformer: {e}")
        embedder = None

# ===================== HTML ===========================
# HTML (supports base64 audio URIs)
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
            font-size: 2rem;
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
            border: 1px solid #333;
            border-radius: 8px;
            background: #1a1a1a;
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
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            display: none;
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
            display: none;
            text-align: center;
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
        let audioUnlocked = false;
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
        function unlockAudio() {
            if (!audioUnlocked) {
                const audio = new Audio('/static/silent.mp3');
                audio.play().then(() => {
                    audioUnlocked = true;
                    console.log('Audio context unlocked');
                }).catch(error => {
                    console.error('Failed to unlock audio:', error);
                });
            }
        }
        micButton.onclick = function() {
            if (recognizing) {
                recognition.stop();
            } else {
                responseContainer.style.display = 'none';
                unlockAudio();
                recognition.start();
            }
        };
        sendButton.onclick = function() {
            const text = textInput.value.trim();
            if (text) {
                unlockAudio();
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
                body: JSON.stringify({prompt: input})
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
        }
    </script>
</body>
</html>
'''

# ===================== UTILITY FUNCTIONS =========================
def search_jiosaavn_song(query):
    try:
        response = requests.get(
            f"https://saavn.me/search/songs?query={query}&limit=1",
            timeout=5
        )
        if response.status_code != 200:
            logger.warning(f"JioSaavn API returned status {response.status_code}")
            return None, None
        data = response.json()
        songs = data.get("results", [])
        if not songs:
            logger.info(f"No songs found for query: {query}")
            return None, None
        song = songs[0]
        song_name = song.get("name", "Unknown Song")
        song_url = next((item["url"] for item in song.get("downloadUrl", []) if item["quality"] == "320kbps"), None)
        if song_url:
            logger.info(f"Found song '{song_name}' with URL: {song_url}")
            return song_name, song_url
        else:
            logger.info(f"No 320kbps URL for song '{song_name}'")
            return None, None
    except Exception as e:
        logger.error(f"JioSaavn search error: {str(e)}")
        return None, None

def embed(text):
    if embedder is None:
        load_embedder()
    if embedder is None:
        logger.error("Embedder not loaded, cannot generate embeddings")
        return np.zeros(384)  # Return zero vector if embedder fails
    return embedder.encode(text)

def redis_search(query, threshold=0.8):
    try:
        query_embedding = embed(query)
        keys = r.keys("qa:*")
        best_match = None
        best_score = 0
        for key in keys:
            cached = r.hgetall(key)
            cached_embedding = np.array(json.loads(cached.get("embedding", "[]")))
            if cached_embedding.size == 0:
                continue
            score = np.dot(query_embedding, cached_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding))
            if score > best_score:
                best_score = score
                best_match = cached
        if best_match and best_score >= threshold:
            logger.info(f"Cache hit for query '{query}' with score {best_score}")
            return best_match
        logger.info(f"Cache miss for query '{query}'")
        return None
    except Exception as e:
        logger.error(f"Redis search error: {e}")
        return None

def redis_store(query, response, audio_base64=None, song_url=None):
    key = f"qa:{hash(query)}"
    data = {
        "query": query,
        "response": response,
        "audio": audio_base64,
        "song_url": song_url,
        "embedding": embed(query).tolist()
    }
    try:
        r.hset(key, mapping=data)
        r.expire(key, 24 * 3600)  # Cache for 24 hours
        logger.info(f"Stored response in Redis for query: {query}")
    except redis.exceptions.RedisError as e:
        logger.error(f"Failed to store in Redis: {e}")

# ===================== GEMINI RESPONSE =========================
def gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        if "song" in prompt.lower() or "music" in prompt.lower():
            song_name, song_url = search_jiosaavn_song(prompt)
            if song_url:
                return f"Playing song: {song_name}", None, song_url
        response = model.generate_content(prompt)
        logger.info(f"Gemini generated response for prompt '{prompt}'")
        return response.text.strip().replace("\n", " "), None, None
    except genai.exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Gemini API error: {str(e)}", None, None
    except Exception as e:
        logger.error(f"Unexpected error in Gemini response: {str(e)}")
        return f"Unexpected error: {str(e)}", None, None

# ===================== GOOGLE TTS =============================
def synthesize_speech(text):
    try:
        if not text or len(text.strip()) == 0:
            logger.error("Empty text provided to Google TTS")
            return None
        if len(text) > 5000:
            logger.warning(f"Text too long for Google TTS ({len(text)} chars); truncating")
            text = text[:5000]
        logger.info(f"Synthesizing speech for text (length: {len(text)}): {text[:50]}...")
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config
        )
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        audio_uri = f"data:audio/mpeg;base64,{audio_base64}"
        logger.info(f"Generated base64 audio URI for response (length: {len(audio_base64)} chars)")
        return audio_uri
    except gcloud_exceptions.GoogleAPIError as e:
        logger.error(f"Google TTS error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in Google TTS: {str(e)}")
        return None

# ===================== FLASK ROUTES ============================
@app.before_request
def enforce_https():
    if not app.debug and request.url.startswith('http://'):
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/favicon.ico")
def favicon():
    try:
        return send_from_directory("static", "favicon.ico")
    except Exception as e:
        logger.error(f"Error serving favicon: {str(e)}")
        return "", 204

@app.route("/static/silent.mp3")
def serve_silent_mp3():
    try:
        return send_from_directory("static", "silent.mp3", mimetype="audio/mpeg")
    except Exception as e:
        logger.error(f"Error serving silent.mp3: {str(e)}")
        return jsonify({"error": "Silent audio file not found"}), 404

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
        if response.startswith("Gemini API error") or response.startswith("Unexpected error"):
            logger.error(f"Gemini response error: {response}")
            return jsonify({"error": response}), 500
        logger.info(f"Gemini response for '{prompt}' (length: {len(response)}): {response[:50]}...")
        audio_base64 = None
        if not song_url:
            audio_base64 = synthesize_speech(response)
            if audio_base64 is None:
                logger.warning("No audio generated for response")
                return jsonify({"response": response, "audio": None, "song_url": None}), 200
        redis_store(prompt, response, audio_base64, song_url)
        logger.info(f"Returning new response for prompt: {prompt}, audio: {audio_base64}, song_url: {song_url}")
        return jsonify({"response": response, "audio": audio_base64, "song_url": song_url})
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/test_tts", methods=["POST"])
def test_tts():
    try:
        data = request.json
        text = data.get("text", "Hello, this is a test.")
        audio_base64 = synthesize_speech(text)
        if audio_base64:
            logger.info(f"Test TTS successful, audio generated")
            return jsonify({"audio": audio_base64})
        else:
            logger.error("Test TTS failed to generate audio")
            return jsonify({"error": "Failed to generate audio"}), 500
    except Exception as e:
        logger.error(f"Test TTS error: {str(e)}")
        return jsonify({"error": f"Test TTS error: {str(e)}"}), 500

# ===================== RUN APP ================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
