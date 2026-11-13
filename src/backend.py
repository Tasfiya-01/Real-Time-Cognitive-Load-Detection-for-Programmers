from flask import Flask, jsonify, render_template_string, request
import google.generativeai as genai
import joblib
import numpy as np
import random
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import skew, kurtosis

app = Flask(__name__)

# Gemini API setup
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini = genai.GenerativeModel('gemini-2.0-flash')

# Model load
model  = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# STEW Dataset load
print("Loading STEW dataset...")
dataset = sio.loadmat('data/dataset.mat')
labels  = sio.loadmat('data/class_012.mat')
eeg_data     = dataset['dataset']
class_labels = labels['class_012'].flatten()
data_T = np.transpose(eeg_data, (2, 1, 0))

SEGMENT_LEN = 640
FS = 128
LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
print("Dataset loaded!")

def bandpass_filter(segment, lowcut=0.5, highcut=50.0, fs=128):
    from scipy import signal
    nyq = 0.5 * fs
    b, a = signal.butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    filtered = np.zeros_like(segment)
    for ch in range(segment.shape[1]):
        filtered[:, ch] = signal.filtfilt(b, a, segment[:, ch])
    return filtered

def normalize(segment):
    normalized = np.zeros_like(segment)
    for ch in range(segment.shape[1]):
        ch_data = segment[:, ch]
        normalized[:, ch] = (ch_data - np.mean(ch_data)) / (np.std(ch_data) + 1e-8)
    return normalized

def extract_features(segment):
    features = []
    for ch in range(segment.shape[1]):
        ch_data = segment[:, ch]
        freqs, psd = welch(ch_data, fs=FS, nperseg=256)
        def bp(low, high):
            idx = np.where((freqs >= low) & (freqs < high))[0]
            return np.mean(psd[idx])
        features.extend([
            bp(0.5,4), bp(4,8), bp(8,13), bp(13,30), bp(30,50),
            bp(4,8)/(bp(8,13)+1e-8), bp(13,30)/(bp(8,13)+1e-8), bp(4,8)/(bp(13,30)+1e-8),
            np.mean(ch_data), np.std(ch_data), np.var(ch_data),
            skew(ch_data), kurtosis(ch_data), np.sqrt(np.mean(ch_data**2))
        ])
    return np.array(features)

def get_load_level():
    subject_idx = random.randint(0, 44)
    segment_idx = random.randint(0, data_T.shape[1]//SEGMENT_LEN - 1)
    start = segment_idx * SEGMENT_LEN
    segment = data_T[subject_idx, start:start+SEGMENT_LEN, :]
    segment = normalize(bandpass_filter(segment))
    features = extract_features(segment)
    prediction = model.predict(scaler.transform([features]))[0]
    load_level = LABELS[int(prediction)]
    print(f"Subject {subject_idx+1} → {load_level}")
    return load_level

def get_suggestion(load_level):
    try:
        response = gemini.generate_content(
            f"A software developer has {load_level} cognitive load. "
            f"Give ONE short helpful suggestion in max 15 words. Only the suggestion, nothing else."
        )
        suggestion = response.text.strip()
        print(f"Gemini: {suggestion}")
        return suggestion
    except Exception as e:
        print(f"Gemini error: {e}")
        defaults = {
            'Low': 'Great focus! Keep it up.',
            'Medium': 'Try breaking the problem into smaller steps.',
            'High': 'Take a 5-minute break to refresh your mind.'
        }
        return defaults[load_level]

def get_code_suggestion(code):
    try:
        response = gemini.generate_content(
            f"Programmer struggling with: {code}. "
            f"Give ONE specific helpful tip in max 15 words. Only the tip, nothing else."
        )
        return response.text.strip()
    except Exception as e:
        return random.choice([
            "Break this into smaller functions.",
            "Add comments to clarify your thinking.",
            "Try a simpler approach first.",
        ])

@app.route('/dashboard')
def dashboard():
    with open('src/dashboard.html', 'r', encoding='utf-8') as f:
        html = f.read()
    return render_template_string(html)

@app.route('/')
def home():
    return jsonify({'status': 'running', 'message': 'Cognitive Load Detection Backend'})

@app.route('/predict')
def predict():
    load_level = get_load_level()
    suggestion = get_suggestion(load_level)
    return jsonify({'load_level': load_level, 'suggestion': suggestion, 'status': 'success'})

@app.route('/code_suggest')
def code_suggest():
    code = request.args.get('code', 'unknown code')
    return jsonify({'code': code, 'suggestion': get_code_suggestion(code)})

@app.route('/status')
def status():
    return jsonify({'model': 'KNN', 'accuracy': '77.78%', 'status': 'active'})

if __name__ == '__main__':
    print("="*50)
    print("  Cognitive Load Backend Starting...")
    print("  URL: http://localhost:5000")
    print("="*50)
    app.run(port=5000, debug=False)