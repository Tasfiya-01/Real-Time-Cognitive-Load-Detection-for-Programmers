from flask import Flask, jsonify, render_template_string, request
import joblib
import numpy as np
import random
import json
import urllib.request
import requests
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import antropy as ant
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_complexity import analyze_code_complexity

app = Flask(__name__)

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

FS           = 128
SEGMENT_LEN  = 640
LABELS       = {0: 'Low', 1: 'Medium', 2: 'High'}
suggestion_cache = {}
last_result  = {'code_hash': '', 'load_level': 'Low', 'complexity': None,
                'cached_suggestion': 'Keep going!', 'last_suggestion_level': ''}
eye_data     = {'blink_rate': 0, 'eye_load': 'Low'}

# ---------------- Ollama config ----------------
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 10  # seconds


def get_load_from_code(code_text):
    complexity = analyze_code_complexity(code_text)
    load_level = complexity['level']
    print(f"Code complexity: {load_level} | "
          f"CC: {complexity['cyclomatic_complexity']}, "
          f"Loops: {complexity['loop_count']}, "
          f"Conditions: {complexity['condition_count']}")
    return load_level, complexity

def get_suggestion(load_level, code=''):
    cache_key = f"{load_level}_{code[:50]}"
    if cache_key in suggestion_cache:
        return suggestion_cache[cache_key]

    suggestions = {
        'Low': [
            'Great focus! Keep writing clean code.',
            'Good momentum! Continue with the current approach.',
            'Code looks simple and readable. Well done!',
            'Nice work! Maintain this clean coding style.',
        ],
        'Medium': [
            'Consider breaking this into smaller functions.',
            'Add comments to explain the logic here.',
            'Review the loop conditions for edge cases.',
            'Try using more descriptive variable names.',
            'Consider adding input validation here.',
        ],
        'High': [
            'Take a 5-minute break to refresh your mind.',
            'This is complex — try simplifying the logic first.',
            'Consider using built-in functions instead.',
            'Break this function into smaller pieces.',
            'Add error handling to make code more robust.',
            'Try explaining the problem out loud first.',
            'Consider using a simpler algorithm approach.',
        ]
    }

    result = random.choice(suggestions.get(load_level, ['Keep going!']))
    suggestion_cache[cache_key] = result
    return result

def get_code_suggestion(code, load_level, complexity_info=None):
    """Rule-based fallback suggestion (used if Ollama is unavailable)."""
    cache_key = f"code_{load_level}_{code[:30]}"
    if cache_key in suggestion_cache:
        return suggestion_cache[cache_key]

    cc    = complexity_info.get('cyclomatic_complexity', 1) if complexity_info else 1
    loops = complexity_info.get('loop_count', 0) if complexity_info else 0

    if load_level == 'High':
        if loops >= 2:
            suggestions = [
                "Consider reducing nested loops for better performance.",
                "Try breaking nested loops into separate functions.",
                "Use list comprehension to simplify nested loops.",
            ]
        else:
            suggestions = [
                "Break this complex function into smaller parts.",
                "Add comments to explain each logical step.",
                "Consider simplifying the conditional logic.",
            ]
    elif load_level == 'Medium':
        suggestions = [
            "Add a docstring to explain what this function does.",
            "Consider edge cases in your current logic.",
            "Review variable names for better readability.",
        ]
    else:
        suggestions = [
            "Good code structure! Consider adding type hints.",
            "Clean and simple — well done!",
            "Consider adding unit tests for this function.",
        ]

    result = random.choice(suggestions)
    suggestion_cache[cache_key] = result
    return result


def get_ollama_suggestion(code, load_level, complexity_info=None):
    """
    Ask the local Ollama (llama3.2) model for a short, real suggestion
    based on the code and its measured complexity. Falls back to the
    rule-based suggestion if Ollama is unreachable, slow, or errors out.
    """
    cache_key = f"ollama_{load_level}_{code[:30]}"
    if cache_key in suggestion_cache:
        return suggestion_cache[cache_key]

    cc    = complexity_info.get('cyclomatic_complexity', 1) if complexity_info else 1
    loops = complexity_info.get('loop_count', 0) if complexity_info else 0

    prompt = (
        "You are a concise coding assistant embedded in a cognitive-load "
        "monitoring tool. Given the code below and its measured complexity, "
        "give exactly ONE short, actionable suggestion (max 15 words) to "
        "help the programmer reduce cognitive load or improve the code. "
        "Do not repeat the code back, do not explain, just give the single suggestion.\n\n"
        f"Cognitive load level: {load_level}\n"
        f"Cyclomatic complexity: {cc}\n"
        f"Loop count: {loops}\n\n"
        f"Code:\n{code}\n\n"
        "Suggestion:"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 40}
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        suggestion = data.get('response', '').strip()

        # Clean up: keep it to a single line, strip quotes/leading dashes
        suggestion = suggestion.split('\n')[0].strip().strip('"').strip()
        suggestion = suggestion.lstrip('-•* ').strip()

        if not suggestion:
            raise ValueError("Empty response from Ollama")

        suggestion_cache[cache_key] = suggestion
        return suggestion

    except Exception as e:
        print(f"[Ollama] Falling back to rule-based suggestion — reason: {e}")
        return get_code_suggestion(code, load_level, complexity_info)


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

@app.route('/predict')
def predict():
    global last_result
    code = request.args.get('code', '')

    if code and len(code.strip()) > 5 and len(code) < 2000:
        code_hash = str(hash(code.strip()))
        if code_hash == last_result['code_hash']:
            # Same code as last time — reuse everything, no new Ollama call
            load_level      = last_result['load_level']
            complexity_info = last_result['complexity']
            suggestion      = last_result.get('cached_suggestion', 'Keep going!')
        else:
            # Code changed — recompute load level AND get a fresh Ollama suggestion
            load_level, complexity_info = get_load_from_code(code)
            suggestion = get_ollama_suggestion(code, load_level, complexity_info)
            last_result['code_hash']             = code_hash
            last_result['load_level']            = load_level
            last_result['complexity']            = complexity_info
            last_result['cached_suggestion']     = suggestion
            last_result['last_suggestion_level'] = load_level
    else:
        load_level      = 'Low'
        complexity_info = None
        suggestion      = last_result.get('cached_suggestion', 'Keep going!')

    return jsonify({
        'load_level':  load_level,
        'suggestion':  suggestion,
        'complexity':  complexity_info,
        'eye_status':  eye_data,
        'model':       'XGBoost',
        'accuracy':    '82.59%',
        'status':      'success'
    })

@app.route('/code_suggest')
def code_suggest():
    code = request.args.get('code', 'unknown code')
    if len(code) > 2000:
        code = code[:2000]
    complexity = analyze_code_complexity(code)
    load_level = complexity['level']
    suggestion = get_ollama_suggestion(code, load_level, complexity)
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
    return jsonify({'status': 'success', 'level': level,
                    'message': f'Profile set to {level}'})

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

@app.route('/update_eye', methods=['POST'])
def update_eye():
    global eye_data
    eye_data = request.get_json()
    return jsonify({'status': 'ok'})

@app.route('/eye_status')
def eye_status_route():
    return jsonify(eye_data)

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
    print("  Suggestion engine: Ollama (llama3.2) with rule-based fallback")
    print("  URL: http://localhost:5000")
    print("="*50)
    app.run(port=5000, debug=False)