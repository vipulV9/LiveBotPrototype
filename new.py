import os
import time
import json
import glob
import numpy as np
import requests
import tempfile
from flask import Flask, render_template_string, request, jsonify, send_from_directory, redirect
import base64
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import google.generativeai as genai
from google.cloud import texttospeech
from google.api_core import exceptions as gcloud_exceptions
@@ -26,47 +25,26 @@
# ===================== CONFIGURE LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# ===================== VALIDATE ENV VARIABLES =================
# ===================== CONFIGURE GEMINI =====================
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

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as temp:
        json.dump(creds_dict, temp)
        temp_path = temp.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
    logger.info(f"Google credentials validated (client_email: {creds_dict['client_email']}, project_id: {creds_dict['project_id']})")
except json.JSONDecodeError:
    logger.error("Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS")
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS must be valid JSON string")


# ===================== CONFIGURE GOOGLE TTS =================
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    raise ValueError("Set GOOGLE_APPLICATION_CREDENTIALS env var")

try:
    creds_dict = json.loads(google_creds)
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
@@ -84,7 +62,7 @@
    raise

# ===================== SENTENCE TRANSFORMERS =================
embedder = None
embedder = None  # Lazy-load to save memory
def load_embedder():
    global embedder
    try:
@@ -95,17 +73,9 @@ def load_embedder():
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
app = Flask(_name_)

# ===================== HTML ===========================
# HTML (supports base64 audio URIs)
HTML = '''
<!DOCTYPE html>
<html>
@@ -118,6 +88,7 @@ def load_embedder():
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: #000;
            color: #e0e0e0;
@@ -128,36 +99,42 @@ def load_embedder():
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
@@ -172,25 +149,30 @@ def load_embedder():
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
@@ -201,10 +183,12 @@ def load_embedder():
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
@@ -215,25 +199,30 @@ def load_embedder():
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
@@ -246,34 +235,40 @@ def load_embedder():
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
@@ -287,26 +282,33 @@ def load_embedder():
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
@@ -320,8 +322,8 @@ def load_embedder():

        let recognizing = false;
        let recognition;
        let audioUnlocked = false;

        // Initialize speech recognition
        if ('webkitSpeechRecognition' in window) {
            recognition = new webkitSpeechRecognition();
        } else if ('SpeechRecognition' in window) {
@@ -332,70 +334,65 @@ def load_embedder():
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
                showError(Microphone error: ${event.error});
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
        // Microphone button click
        micButton.onclick = function() {
            if (recognizing) {
                recognition.stop();
            } else {
                responseContainer.style.display = 'none';
                recognition.start();
            }
        }
        };

        function tryPlayAudio(audio, retries = 3, delay = 500) {
            if (retries === 0) {
                audioPlayerDiv.innerHTML += `
                    <button onclick="playAudioManually()">Play Audio</button>
                `;
                showError('Autoplay blocked. Click the Play button to hear the response.');
                return;
        // Send button click
        sendButton.onclick = function() {
            const text = textInput.value.trim();
            if (text) {
                processUserInput(text);
                textInput.value = '';
            }
            audio.play().then(() => {
                audioUnlocked = true;
            }).catch(error => {
                console.error('Autoplay attempt failed:', error);
                setTimeout(() => tryPlayAudio(audio, retries - 1, delay), delay);
            });
        }
        };

        function playAudioManually() {
            const audio = document.getElementById('audio-player');
            audio.play().then(() => {
                audioUnlocked = true;
            }).catch(error => {
                showError(`Audio playback error: ${error.message}`);
            });
        }
        // Enter key in text input
        textInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                sendButton.click();
            }
        });

        function processUserInput(input) {
            userQuestionDiv.textContent = `You: ${input}`;
            userQuestionDiv.textContent = You: ${input};
            statusDiv.textContent = 'Getting response...';

            // Send to server for processing
            fetch("/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
@@ -406,61 +403,33 @@ def load_embedder():
                if (data.error) {
                    showError(data.error);
                } else {
                    aiResponseDiv.textContent = `Assistant: ${data.response}`;
                    aiResponseDiv.textContent = Assistant: ${data.response};
                    audioPlayerDiv.innerHTML = '';
                    if (data.audio || data.song_url) {
                        const audioSrc = data.audio || data.song_url;
                    if (data.audio) {
                        audioPlayerDiv.innerHTML = `
                            <audio id="audio-player" controls>
                                <source src="${audioSrc}" type="audio/mpeg">
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
                        const audio = document.getElementById('audio-player');
                        audio.load();
                        if (audioUnlocked) {
                            tryPlayAudio(audio);
                        } else {
                            audioPlayerDiv.innerHTML += `
                                <button onclick="playAudioManually()">Play Audio</button>
                            `;
                            showError('Click the Play button to hear the response.');
                        }
                    }
                    responseContainer.style.display = 'flex';
                    statusDiv.textContent = 'Choose voice or text input';
                }
            })
            .catch(error => {
                showError(`Error: ${error}`);
                showError(Error: ${error});
            });
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

        function showError(message) {
            statusDiv.textContent = 'Error occurred';
            errorContainer.textContent = message;
@@ -485,10 +454,6 @@ def search_jiosaavn_song(query):
            song_name = song.get("name", "Unknown Song")
            song_url = next((item["url"] for item in song.get("downloadUrl", []) if item["quality"] == "320kbps"), None)
            if song_url:
                test_response = requests.head(song_url, timeout=5, allow_redirects=True)
                if test_response.status_code != 200:
                    logger.warning(f"Song URL {song_url} is not accessible (status: {test_response.status_code})")
                    return song_name, None
                logger.info(f"Found song '{song_name}' with URL: {song_url}")
                return song_name, song_url
            else:
@@ -547,21 +512,17 @@ def redis_search(query, threshold=0.8):
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
def redis_store(query, response, audio_base64=None, song_url=None):
    key = f"qa:{hash(query)}"
    data = {
        "query": query,
        "response": response,
        "audio": audio_path,
        "audio": audio_base64,
        "song_url": song_url,
        "embedding": embed(query).tolist()
    }
@@ -571,17 +532,6 @@ def redis_store(query, response, audio_path=None, song_url=None):
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
@@ -598,21 +548,16 @@ def gemini_response(prompt):
        response = model.generate_content(prompt)
        logger.info(f"Gemini generated response for prompt '{prompt}'")
        return response.text.strip().replace("\n", " "), None, None
    except Exception as e:
    except genai.exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Gemini API error: {str(e)}", None, None
    except Exception as e:
        logger.error(f"Unexpected error in Gemini response: {str(e)}")
        return f"Unexpected error: {str(e)}", None, None

# ===================== GOOGLE TTS =============================
def synthesize_speech(text, filename=None):
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
@@ -627,41 +572,24 @@ def synthesize_speech(text, filename=None):
            voice=voice,
            audio_config=audio_config
        )
        if len(response.audio_content) < 1000:
            logger.error(f"Audio content too small ({len(response.audio_content)} bytes)")
            return None

        filename = filename or f"reply_{int(time.time() * 1000)}.mp3"
        filepath = os.path.join(AUDIO_FOLDER, filename)
        with open(filepath, "wb") as out:
            out.write(response.audio_content)
            out.flush()
            os.fsync(out.fileno())
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            if file_size < 1000:
                logger.error(f"Audio file {filepath} is too small ({file_size} bytes)")
                return None
            os.chmod(filepath, 0o644)
            logger.info(f"Generated audio at {filepath} (size: {file_size} bytes)")
            return f"/static/audio/{filename}"
        else:
            logger.error(f"File existence check failed for {filepath}")
            return None
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        audio_uri = f"data:audio/mpeg;base64,{audio_base64}"
        logger.info(f"Generated base64 audio URI for response (length: {len(audio_base64)} chars)")
        return audio_uri
    except gcloud_exceptions.InvalidArgument as e:
        logger.error(f"Invalid input for Google TTS: {str(e)}")
        return None
    except gcloud_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied for Google TTS: {str(e)}")
        return None
    except gcloud_exceptions.GoogleAPIError as e:
        logger.error(f"Google TTS API error: {str(e)}", exc_info=True)
        logger.error(f"Google TTS error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in Google TTS: {str(e)}", exc_info=True)
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
@@ -673,84 +601,41 @@ def favicon():
        return send_from_directory("static", "favicon.ico")
    return "", 204

@app.route("/static/silent.mp3")
def serve_silent_mp3():
    try:
        return send_from_directory("static", "silent.mp3", mimetype="audio/mpeg")
    except Exception as e:
        logger.error(f"Error serving silent.mp3: {str(e)}")
        return jsonify({"error": "Silent audio file not found"}), 404

@app.route("/static/audio/<filename>")
def serve_audio(filename):
    try:
        filepath = os.path.join(AUDIO_FOLDER, filename)
        if not os.path.exists(filepath):
            logger.error(f"Audio file {filepath} not found")
            return jsonify({"error": "Audio file not found"}), 404
        return send_from_directory(AUDIO_FOLDER, filename, mimetype="audio/mpeg")
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {str(e)}")
        return jsonify({"error": "Audio file not found"}), 404

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
        if response.startswith("Gemini API error") or response.startswith("Unexpected error"):
            return jsonify({"error": response}), 500

        logger.info(f"Gemini response for '{prompt}' (length: {len(response)}): {response[:50]}...")

        audio_path = None
        audio_base64 = None
        if not song_url:
            audio_path = synthesize_speech(response)
            if audio_path is None:
            audio_base64 = synthesize_speech(response)
            if audio_base64 is None:
                logger.warning("No audio generated for response")
                return jsonify({"response": response, "audio": None, "song_url": None}), 200

        redis_store(prompt, response, audio_path, song_url)
        cleanup_audio()
        redis_store(prompt, response, audio_base64, song_url)

        logger.info(f"Returning new response for prompt: {prompt}, audio: {audio_path}, song_url: {song_url}")
        return jsonify({"response": response, "audio": audio_path, "song_url": song_url})
        return jsonify({"response": response, "audio": audio_base64, "song_url": song_url})
    except Exception as e:
        logger.error(f"Server error in /chat: {str(e)}", exc_info=True)
        logger.error(f"Server error: {str(e)}")
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
if _name_ == "_main_":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
