from flask import Flask, jsonify, render_template_string, request
import joblib
import numpy as np
import random
import json
import urllib.request
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import antropy as ant
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_complexity import analyze_code_complexity

app = Flask(__name__)

GEMINI_KEY = "YOUR_GEMINI_KEY_HERE"

print("Loading model...")
model  = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler_v3.pkl')
le     = joblib.load('models/label_encoder.pkl')
print("Model loaded!")

print("Loading STEW dataset...")
dataset      = sio.loadmat('data/dataset.mat')
labels_mat   = sio.loadmat('data/class_012.mat')
eeg_data     = dataset['dataset']
class_labels = labels_mat['class_012'].flatten()
data_T       = np.transpose(eeg_data, (2, 1, 0))
print("Dataset loaded!")

FS          = 128
SEGMENT_LEN = 640
LABELS      = {0: 'Low', 1: 'Medium', 2: 'High'}

suggestion_cache = {}

def gemini_call(prompt):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
    )
    data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}]
    }).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        res = json.loads(r.read())
        return res['candidates'][0]['content']['parts'][0]['text']

def get_load_from_code(code_text):
    complexity = analyze_code_complexity(code_text)
    load_level = complexity['level']
    print(f"Code complexity: {load_level} | "
          f"CC: {complexity['cyclomatic_complexity']}, "
          f"Loops: {complexity['loop_count']}, "
          f"Conditions: {complexity['condition_count']}")
    return load_level, complexity

def get_suggestion(load_level, code=''):
    cache_key = f"{load_level}_{code[:30]}"
    if cache_key in suggestion_cache:
        return suggestion_cache[cache_key]

    try:
        code_context = f"Code: {code[:150]}. " if code else ""
        prompt = (
            f"Developer has {load_level} cognitive load. "
            f"{code_context}"
            f"Give ONE specific tip in max 15 words. Only the tip."
        )
        result = gemini_call(prompt)
        suggestion_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"Gemini error: {e}")
        defaults = {
            'Low':    'Great focus! Keep it up.',
            'Medium': 'Try breaking the problem into smaller steps.',
            'High':   'Take a 5-minute break to refresh your mind.'
        }
        return defaults.get(load_level, 'Keep going!')

def get_code_suggestion(code, load_level, complexity_info=None):
    cache_key = f"code_{load_level}_{code[:30]}"
    if cache_key in suggestion_cache:
        return suggestion_cache[cache_key]

    try:
        context = ""
        if complexity_info:
            context = (
                f"CC: {complexity_info.get('cyclomatic_complexity', 'N/A')}, "
                f"Loops: {complexity_info.get('loop_count', 0)}. "
            )
        prompt = (
            f"Programmer has {load_level} cognitive load. "
            f"{context}Code: {code[:200]}. "
            f"Give ONE specific helpful tip in max 15 words. Only the tip."
        )
        result = gemini_call(prompt)
        suggestion_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"Gemini error: {e}")
        return random.choice([
            "Break this into smaller functions.",
            "Add comments to clarify your thinking.",
            "Try a simpler approach first.",
        ])

eye_status = {'blink_rate': 0, 'eye_load': 'Low'}

@app.route('/eye_status', methods=['GET', 'POST'])
def update_eye_status():
    global eye_status
    if request.method == 'POST':
        eye_status['blink_rate'] = request.json.get('blink_rate', 0)
        eye_status['eye_load']   = request.json.get('eye_load', 'Low')
    return jsonify(eye_status)

@app.route('/dashboard')
def dashboard():
    with open('src/dashboard.html', 'r', encoding='utf-8') as f:
        html = f.read()
    return render_template_string(html)

@app.route('/')
def home():
    return jsonify({
        'status':   'running',
        'message':  'Cognitive Load Detection Backend v2',
        'model':    'XGBoost',
        'accuracy': '82.59%'
    })

last_result = {'code_hash': '', 'load_level': 'Low', 'complexity': None}

@app.route('/predict')
def predict():
    global last_result
    code = request.args.get('code', '')

    if code and len(code.strip()) > 5 and len(code) < 2000:
        code_hash = str(hash(code.strip()))
        if code_hash == last_result['code_hash']:
            load_level      = last_result['load_level']
            complexity_info = last_result['complexity']
        else:
            load_level, complexity_info = get_load_from_code(code)
            last_result = {
                'code_hash':  code_hash,
                'load_level': load_level,
                'complexity': complexity_info
            }
    else:
        load_level      = 'Low'
        complexity_info = None

    suggestion = get_suggestion(load_level, code[:200] if code else '')

    return jsonify({
        'load_level':  load_level,
        'suggestion':  suggestion,
        'complexity':  complexity_info,
        'eye_status':  eye_status,
        'model':       'XGBoost',
        'accuracy':    '82.59%',
        'status':      'success'
    })

@app.route('/code_suggest')
def code_suggest():
    code       = request.args.get('code', 'unknown code')
    if len(code) > 2000:
        code = code[:2000]
    complexity = analyze_code_complexity(code)
    load_level = complexity['level']
    suggestion = get_code_suggestion(code, load_level, complexity)
    return jsonify({
        'code':       code,
        'load_level': load_level,
        'complexity': complexity,
        'suggestion': suggestion
    })

@app.route('/set_profile')
def set_profile():
    level = request.args.get('level', 'Intermediate')
    if level not in ['Beginner', 'Intermediate', 'Expert']:
        return jsonify({'error': 'Invalid level'}), 400
    from personalized_threshold import save_profile
    save_profile(level)
    last_result['code_hash'] = ''
    return jsonify({'status': 'success', 'level': level, 'message': f'Profile set to {level}'})

@app.route('/predict_personalized')
def predict_personalized():
    from personalized_threshold import analyze_with_profile
    code = request.args.get('code', '')
    if not code or len(code.strip()) < 5:
        return jsonify({'load_level': 'Low', 'complexity': None})
    result     = analyze_with_profile(code)
    load_level = result['level']
    suggestion = get_suggestion(load_level, code[:200])
    return jsonify({
        'load_level':  load_level,
        'suggestion':  suggestion,
        'complexity':  result,
        'user_level':  result['user_level'],
        'status':      'success'
    })

@app.route('/status')
def status():
    return jsonify({
        'model':    'XGBoost',
        'accuracy': '82.59%',
        'features': 252,
        'status':   'active'
    })

if __name__ == '__main__':
    print("="*50)
    print("  Cognitive Load Backend v2 Starting...")
    print("  Model: XGBoost (82.59% accuracy)")
    print("  URL: http://localhost:5000")
    print("="*50)
    app.run(port=5000, debug=False)