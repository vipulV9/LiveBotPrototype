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
from PIL import Image
import io
import base64  # For decoding Craiyon base64 images
from craiyon import Craiyon  # Ensure this is in your imports

# ===================== LOAD ENV VARIABLES =====================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# ===================== CONFIGURE LOGGING =====================
logging.basicConfig(
    level=logging.DEBUG,
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

try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Configured genai with GEMINI_API_KEY")
except Exception as e:
    logger.warning(f"genai.configure failed: {e} (continuing with fallback)")

# ===================== CONFIGURE GOOGLE CLOUD (for TTS only) =================
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    raise ValueError("Set GOOGLE_APPLICATION_CREDENTIALS env var")

# ===================== REDIS ================================
REDIS_HOST = "redis-12173.crce217.ap-south-1-1.ec2.redns.redis-cloud.com"
REDIS_PORT = 12173
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
    logger.error(f"Redis connection error: {e}", exc_info=True)
    raise

# ===================== SENTENCE TRANSFORMERS =================
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Initialized SentenceTransformer model 'all-MiniLM-L6-v2'")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer: {e}", exc_info=True)
    embedder = None

# ===================== FLASK APP ===========================
app = Flask(__name__)
AUDIO_FOLDER = "static/audio"
IMAGE_FOLDER = "static/images"
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ===================== HTML ===========================
# Modified to remove aspect ratio selection since Craiyon only supports square images
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant + Image Gen</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background:#000; color:#e0e0e0; min-height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:20px;}
        .container { width:100%; max-width:900px; text-align:center; }
        h1 { font-size:2.2rem; margin-bottom:1rem; color:#fff; font-weight:300; }
        .input-section { display:flex; flex-direction:column; align-items:center; margin-bottom:1.5rem; }
        .input-options { display:flex; justify-content:center; align-items:center; gap:30px; margin-bottom:20px; flex-wrap:wrap; }
        .mic-container { position:relative; width:100px; height:100px; }
        .mic-button { width:100px; height:100px; border-radius:50%; background:#1a1a1a; border:2px solid #333; display:flex; align-items:center; justify-content:center; font-size:2.5rem; cursor:pointer; transition:all .2s; }
        .mic-button:hover { background:#222; transform:scale(1.02); }
        .mic-button.recording { animation:pulse 1.5s infinite; border-color:#555; }
        @keyframes pulse { 0%{ box-shadow:0 0 0 0 rgba(255,255,255,.06);} 70%{ box-shadow:0 0 0 20px rgba(255,255,255,0);} 100%{ box-shadow:0 0 0 0 rgba(255,255,255,0);} }
        .text-input { width:300px; padding:10px 12px; background:#1a1a1a; border:1px solid #333; border-radius:8px; color:#e0e0e0; }
        .text-input:focus { outline:none; border-color:#4a90e2; }
        .send-button { padding:10px 20px; background:#1a1a1a; border:1px solid #333; border-radius:8px; color:#e0e0e0; cursor:pointer; }
        .send-button:hover { background:#222; border-color:#555; }
        .divider { width:1px; height:80px; background:#333; }
        .status { margin-top:12px; font-size:1rem; height:24px; color:#aaa; }
        .response-container { background:#111; border:1px solid #333; border-radius:12px; padding:18px; margin-top:1.5rem; min-height:150px; display:flex; flex-direction:column; gap:12px; align-items:stretch; }
        .user-question, .ai-response { padding:12px; border-radius:8px; background:#1a1a1a; text-align:left; }
        .user-question { border-left:4px solid #4a90e2; align-self:flex-end; max-width:80%; }
        .ai-response { border-left:4px solid #5cb85c; align-self:flex-start; max-width:80%; }
        .audio-player audio, .image-preview img { width:100%; border-radius:8px; max-width:512px; }
        .image-controls { display:flex; gap:8px; align-items:center; justify-content:center; margin-top:12px; flex-wrap:wrap; }
        .small-input { padding:8px; border-radius:8px; background:#1a1a1a; border:1px solid #333; color:#e0e0e0; }
        .error { color:#ff6b6b; padding:10px; background:#1a1a1a; border:1px solid #ff6b6b; border-radius:8px; margin-top:12px; display:none;}
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Assistant + Image Generator</h1>
        <div class="input-section">
            <div class="input-options">
                <div class="mic-container">
                    <div id="mic-button" class="mic-button">🎤</div>
                </div>
                <div class="divider"></div>
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <input id="text-input" class="text-input" placeholder="Type your message">
                    <div style="margin-top:8px;">
                        <button id="send-button" class="send-button">Send</button>
                    </div>
                </div>
            </div>
            <div style="margin-top:8px; width:100%; display:flex; flex-direction:column; align-items:center;">
                <div class="image-controls">
                    <input id="image-prompt" class="small-input" placeholder="Enter image prompt (e.g. A red sports car on a neon street)" style="width:420px;">
                    <button id="generate-image" class="send-button">Generate Image</button>
                </div>
            </div>
        </div>
        <div id="status" class="status">Choose voice, text or image input</div>
        <div id="response-container" class="response-container" style="display:none;">
            <div id="user-question" class="user-question"></div>
            <div id="ai-response" class="ai-response"></div>
            <div id="audio-player"></div>
            <div id="image-preview" class="image-preview"></div>
        </div>
        <div id="error-container" class="error"></div>
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
        const imagePreviewDiv = document.getElementById('image-preview');
        const errorContainer = document.getElementById('error-container');
        const imagePromptInput = document.getElementById('image-prompt');
        const generateImageBtn = document.getElementById('generate-image');
        let recognition, recognizing = false;
        if ('webkitSpeechRecognition' in window) {
            recognition = new webkitSpeechRecognition();
        } else if ('SpeechRecognition' in window) {
            recognition = new SpeechRecognition();
        }
        if (recognition) {
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            recognition.onstart = () => {
                recognizing = true;
                micButton.classList.add('recording');
                statusDiv.textContent = 'Listening...';
                errorContainer.style.display = 'none';
            };
            recognition.onend = () => {
                recognizing = false;
                micButton.classList.remove('recording');
                statusDiv.textContent = 'Processing...';
            };
            recognition.onresult = (event) => {
                const t = event.results[0][0].transcript;
                processUserInput(t);
            };
            recognition.onerror = (e) => {
                recognizing = false;
                micButton.classList.remove('recording');
                showError('Microphone error: ' + e.error);
            };
        } else {
            showError('Browser does not support SpeechRecognition API.');
        }
        micButton.onclick = () => {
            if (!recognition) return;
            if (recognizing) recognition.stop();
            else {
                responseContainer.style.display = 'none';
                recognition.start();
            }
        };
        sendButton.onclick = () => {
            const text = textInput.value.trim();
            if (!text) return;
            processUserInput(text);
            textInput.value = '';
        };
        textInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') sendButton.click();
        });
        function processUserInput(s) {
            userQuestionDiv.textContent = `You: ${s}`;
            statusDiv.textContent = 'Getting response...';
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: s})
            }).then(r => r.json()).then(data => {
                if (data.error) showError(data.error);
                else {
                    aiResponseDiv.textContent = `Assistant: ${data.response || ''}`;
                    audioPlayerDiv.innerHTML = '';
                    imagePreviewDiv.innerHTML = '';
                    if (data.audio) {
                        audioPlayerDiv.innerHTML = `<audio controls autoplay><source src="${data.audio}" type="audio/mpeg">Your browser does not support the audio element.</audio>`;
                    }
                    if (data.song_url) {
                        audioPlayerDiv.innerHTML = `<audio controls autoplay><source src="${data.song_url}" type="audio/mpeg">Your browser does not support the audio element.</audio>`;
                    }
                    if (data.image_url) {
                        imagePreviewDiv.innerHTML = `<img src="${data.image_url}" alt="Generated image">`;
                    }
                    responseContainer.style.display = 'flex';
                    statusDiv.textContent = 'Choose voice, text or image input';
                }
            }).catch(e => showError('Error: ' + e));
        }
        generateImageBtn.onclick = () => {
            const prompt = imagePromptInput.value.trim();
            if (!prompt) { showError('Enter an image prompt'); return; }
            statusDiv.textContent = 'Generating image...';
            errorContainer.style.display = 'none';
            fetch('/generate_image', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt})
            }).then(r => r.json()).then(data => {
                if (data.error) showError(data.error);
                else {
                    responseContainer.style.display = 'flex';
                    userQuestionDiv.textContent = `Image prompt: ${prompt}`;
                    aiResponseDiv.textContent = data.caption ? `Caption: ${data.caption}` : '';
                    audioPlayerDiv.innerHTML = '';
                    imagePreviewDiv.innerHTML = `<img src="${data.image_url}" alt="Generated image">`;
                    statusDiv.textContent = 'Image ready';
                }
            }).catch(e => showError('Error: ' + e));
        };
        function showError(msg) {
            statusDiv.textContent = 'Error occurred';
            errorContainer.textContent = msg;
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
        response = requests.get(url, params=params)
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
        logger.error(f"JioSaavn API error: {str(e)}", exc_info=True)
        return None, None


# ===================== EMBEDDINGS & COSINE SIM ===================
def embed(text):
    try:
        if embedder is None:
            raise ValueError("SentenceTransformer not initialized")
        embedding = embedder.encode(text, convert_to_numpy=True)
        logger.debug(f"Generated embedding for text: {text}")
        return embedding
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}", exc_info=True)
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
        logger.error(f"Redis search error: {e}", exc_info=True)
        return None
    if best_match and best_score >= threshold:
        logger.info(f"Cache hit for query '{query}' with score {best_score}")
        return best_match
    logger.info(f"Cache miss for query '{query}'")
    return None


def redis_store(query, response, audio_path=None, song_url=None, image_url=None, caption=None):
    key = f"qa:{hash(query)}"
    data = {
        "query": query,
        "response": response,
        "audio": audio_path,
        "song_url": song_url,
        "image_url": image_url,
        "caption": caption,
        "embedding": embed(query).tolist()
    }
    try:
        r.setex(key, 24 * 3600, json.dumps(data))
        logger.info(f"Stored query '{query}' in Redis with TTL 24h")
    except redis.exceptions.RedisError as e:
        logger.error(f"Failed to store in Redis: {e}", exc_info=True)


# ===================== CLEANUP ===========================
def cleanup_audio(max_age_hours=24):
    now = time.time()
    for file in glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3")):
        try:
            if os.path.getmtime(file) < now - (max_age_hours * 3600):
                os.remove(file)
                logger.info(f"Deleted old audio file: {file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup audio {file}: {e}", exc_info=True)


def cleanup_images(max_age_hours=72):
    now = time.time()
    for file in glob.glob(os.path.join(IMAGE_FOLDER, "*.png")):
        try:
            if os.path.getmtime(file) < now - (max_age_hours * 3600):
                os.remove(file)
                logger.info(f"Deleted old image file: {file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup image {file}: {e}", exc_info=True)


# ===================== GEMINI / TEXT RESPONSE =========================
def gemini_response(prompt):
    try:
        prompt_lower = prompt.lower()
        if any(keyword in prompt_lower for keyword in ["play", "song", "music"]):
            song_name, song_url = search_jiosaavn_song(prompt)
            if song_name and song_url:
                return f"Playing '{song_name}'", None, song_url, None
            elif song_name:
                return f"Found song '{song_name}' but no playable URL available", None, None, None
            else:
                return "Sorry, I couldn't find that song", None, None, None
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            res = model.generate_content(prompt)
            logger.info(f"Gemini generated response for prompt '{prompt}'")
            return res.text.strip().replace("\n", " "), None, None, None
        except Exception as e:
            logger.error(f"Gemini text generation failed: {e}", exc_info=True)
            return f"Gemini API error: {e}", None, None, None
    except Exception as e:
        logger.error(f"gemini_response exception: {e}", exc_info=True)
        return f"Gemini API error: {e}", None, None, None


# ===================== IMAGE GENERATION (Craiyon) =========================
def generate_image(prompt):
    try:
        logger.info(f"Generating image with Craiyon for prompt: {prompt}")

        # Create Craiyon generator instance
        generator = Craiyon()

        # Generate images - this will return multiple base64 encoded images
        result = generator.generate(prompt)

        if not result or not hasattr(result, 'images') or not result.images:
            logger.error("No images returned by Craiyon")
            return None, "No images were generated. Please try a different prompt."

        # Get the first image (Craiyon returns multiple options)
        try:
            img_data = base64.b64decode(result.images[0])
            image = Image.open(io.BytesIO(img_data))
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return None, f"Failed to process generated image: {str(e)}"

        # Save the image
        timestamp = int(time.time())
        filename = f"img_{timestamp}.png"
        filepath = os.path.join(IMAGE_FOLDER, filename)

        image.save(filepath, "PNG")
        logger.info(f"Saved generated image to {filepath}")

        # Optional: Generate a caption for the image
        caption = None
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-8b")
            response = model.generate_content(f"Describe this image briefly (based on the prompt): {prompt}")
            caption = response.text.strip()
            logger.info(f"Generated caption: {caption}")
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")

        # Return the URL path to the image and caption
        image_url = f"/static/images/{filename}"
        return image_url, caption

    except Exception as e:
        logger.error(f"Image generation error: {str(e)}", exc_info=True)
        return None, f"Image generation failed: {str(e)}"


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
        return f"/{filepath.replace(os.path.sep, '/')}"
    except gcloud_exceptions.GoogleCloudError as e:
        logger.error(f"Google TTS error: {str(e)}", exc_info=True)
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
            return jsonify({"error": "No prompt provided"}), 400

        cached = redis_search(prompt)
        if cached:
            return jsonify({
                "response": cached["response"],
                "audio": cached.get("audio"),
                "song_url": cached.get("song_url"),
                "image_url": cached.get("image_url"),
                "caption": cached.get("caption")
            })

        response, audio_path, song_url, image_url = gemini_response(prompt)
        if response.startswith("Gemini API error"):
            return jsonify({"error": response}), 500

        if not song_url and not image_url:
            audio_path = synthesize_speech(response)
            if audio_path is None:
                redis_store(prompt, response, None, None, None, None)
                return jsonify(
                    {"response": response, "audio": None, "song_url": None, "image_url": None, "caption": None}), 200

        redis_store(prompt, response, audio_path, song_url, image_url, None)
        cleanup_audio()
        cleanup_images()

        return jsonify(
            {"response": response, "audio": audio_path, "song_url": song_url, "image_url": image_url, "caption": None})
    except Exception as e:
        logger.error(f"Server error (chat): {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# Fixed route to use our local generate_image function
@app.route("/generate_image", methods=["POST"])
def generate_image_route():
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Check if we have a cached result for this prompt
        cached = redis_search(prompt)
        if cached and cached.get("image_url"):
            logger.info(f"Using cached image for prompt '{prompt}'")
            return jsonify({
                "image_url": cached["image_url"],
                "caption": cached.get("caption")
            })

        # Generate new image using the Craiyon implementation
        image_url, caption = generate_image(prompt)

        if not image_url:
            return jsonify({"error": caption or "Failed to generate image"}), 500

        # Store the result in Redis
        redis_store(prompt, f"Generated image for: {prompt}", None, None, image_url, caption)

        return jsonify({
            "image_url": image_url,
            "caption": caption
        })
    except Exception as e:
        logger.error(f"Image generation route error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


# ===================== RUN APP ================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
