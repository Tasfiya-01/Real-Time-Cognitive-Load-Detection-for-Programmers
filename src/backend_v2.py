from flask import Flask, jsonify, render_template_string, request
import joblib
import numpy as np
import random
import time
import requests
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_complexity import analyze_code_complexity

app = Flask(__name__)

# ---------------- EEG playback (brain state) ----------------
# No EEG headset is connected. The brain state comes from recorded EEG of
# participant P05 (Peitek et al. dataset), played back one 5-s window every 5 s.
# The model was trained on the other 9 participants (reading code vs rest),
# so P05 is a new, unseen person for the model.
print("Loading EEG model and playback data...")
prog_model = joblib.load('models/prog_model.pkl')
PB = np.load('models/playback_p05.npz')
PB_START = time.time()
EEG_STEP_S = 5
print(f"EEG playback ready: {len(PB['y'])} windows ({len(PB['y']) * EEG_STEP_S / 60:.1f} min)")

# Per-electrode theta/alpha for the head map, read from the same playback features.
# Feature column of channel c, feature j = c*18 + j (theta j=1, alpha j=2); PB['X'] holds the kept columns.
CH_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
_KEEP = list(np.load('models/keep_idx.npy'))
_TH = [_KEEP.index(c * 18 + 1) for c in range(14)]
_AL = [_KEEP.index(c * 18 + 2) for c in range(14)]
_RATIO = np.log10((PB['X'][:, _TH] + 1e-12) / (PB['X'][:, _AL] + 1e-12))     # (windows, 14)
_RATIO_Z = (_RATIO - _RATIO.mean(0)) / (_RATIO.std(0) + 1e-12)               # relative to this session

# Java source of the 32 snippets P05 read (Peitek et al. replication package, CC BY-SA 4.0),
# with cyclomatic complexity measured by lizard. Used by the dashboard's replay mode.
import json
with open('models/snippets.json', encoding='utf-8') as _f:
    SNIPPETS = json.load(_f)
print(f"Snippets loaded: {len(SNIPPETS)}")

suggestion_cache = {}
last_result  = {'code_hash': '', 'load_level': 'Low', 'complexity': None,
                'cached_suggestion': 'Keep going!', 'last_suggestion_level': ''}
# Webcam eye tracker (src/eye_tracking.py) posts here every 5 s. None = no webcam data.
eye_data     = {'blink_rate': None, 'eye_load': None, 'detector': None, 'updated': 0.0}
EYE_STALE_S  = 15   # older eye data is treated as missing

# ---------------- Ollama config ----------------
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 10  # seconds


def current_eeg_state():
    """Brain state for the current playback window: P(engaged) from the programmer model."""
    i = int((time.time() - PB_START) / EEG_STEP_S) % len(PB['y'])
    p = float(prog_model.predict_proba(PB['X'][i:i + 1])[0, 1])
    name = str(PB['snippet'][i])
    reading = int(PB['y'][i]) == 1
    sn = SNIPPETS.get(name, {})
    return {
        # what P05 had on screen in this window: the Java snippet while reading, a fixation cross while resting
        'on_screen':  {'name': name, 'code': sn.get('code', ''), 'cc_max': sn.get('cc_max'),
                       'nloc': sn.get('nloc'), 'loops': sn.get('loops'), 'conditions': sn.get('conditions'),
                       'level': sn.get('level')} if reading else None,
        'source':     'EEG playback (recorded, participant P05, unseen by the model)',
        'window':     i,
        'n_windows':  int(len(PB['y'])),
        'time_s':     i * EEG_STEP_S,
        'p_engaged':  round(p, 3),
        'state':      'Engaged' if p >= 0.5 else 'Resting',
        'true_phase': 'reading code' if int(PB['y'][i]) == 1 else 'rest',
        'snippet':    str(PB['snippet'][i]),
        # theta/alpha per electrode, z-scored over the playback session (head map)
        'channels':   {ch: round(float(_RATIO_Z[i, k]), 2) for k, ch in enumerate(CH_NAMES)}
    }


def get_load_from_code(code_text):
    """Code complexity level (Low/Medium/High) from radon-based analysis."""
    try:
        from personalized_threshold import get_thresholds
        thresholds = get_thresholds()          # profile chosen on the dashboard (default Intermediate)
    except Exception:
        thresholds = None
    complexity = analyze_code_complexity(code_text, thresholds)
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


def get_ollama_suggestion(code, load_level, complexity_info=None, brain_state=None):
    """
    Ask the local Ollama (llama3.2) model for a short, real suggestion
    based on the code, its measured complexity and (if given) the EEG brain state.
    Falls back to the rule-based suggestion if Ollama is unreachable, slow, or errors out.
    """
    cache_key = f"ollama_{load_level}_{code[:30]}"
    if cache_key in suggestion_cache:
        return suggestion_cache[cache_key]

    cc    = complexity_info.get('cyclomatic_complexity', 1) if complexity_info else 1
    loops = complexity_info.get('loop_count', 0) if complexity_info else 0
    brain = brain_state or 'unknown'

    prompt = (
        "You are a concise coding assistant embedded in a cognitive-load "
        "monitoring tool. Given the code below, its measured complexity and the "
        "programmer's EEG brain state, give exactly ONE short, actionable suggestion "
        "(max 15 words) to help the programmer reduce cognitive load or improve the code. "
        "Do not repeat the code back, do not explain, just give the single suggestion.\n\n"
        f"Code complexity level: {load_level}\n"
        f"Cyclomatic complexity: {cc}\n"
        f"Loop count: {loops}\n"
        f"EEG brain state: {brain}\n\n"
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
        'status':        'running',
        'message':       'Cognitive Load Detection Backend v2',
        'code_load':     'radon code complexity',
        'brain_state':   'EEG playback, programmer model (see /eeg_state)'
    })

@app.route('/eeg_state')
def eeg_state():
    """Brain state from EEG playback: probability that the programmer is engaged (reading code) vs resting."""
    return jsonify(current_eeg_state())

@app.route('/predict')
def predict():
    global last_result
    code = request.args.get('code', '')
    eeg  = current_eeg_state()

    code = code[:2000]                     # the extension sends at most 2,000 characters
    if code and len(code.strip()) > 5:
        code_hash = str(hash(code.strip()))
        if code_hash == last_result['code_hash']:
            # Same code as last time — reuse everything, no new Ollama call
            load_level      = last_result['load_level']
            complexity_info = last_result['complexity']
            suggestion      = last_result.get('cached_suggestion', 'Keep going!')
        else:
            # Code changed — recompute complexity AND get a fresh Ollama suggestion
            load_level, complexity_info = get_load_from_code(code)
            suggestion = get_ollama_suggestion(code, load_level, complexity_info, eeg['state'])
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
        'load_level':  load_level,              # code complexity level (Low/Medium/High)
        'load_source': 'code_complexity',
        'suggestion':  suggestion,
        'complexity':  complexity_info,
        'eeg':         eeg,                     # brain state from EEG playback
        'eye_status':  eye_status(),
        'status':      'success'
    })

@app.route('/code_suggest')
def code_suggest():
    code = request.args.get('code', 'unknown code')
    if len(code) > 2000:
        code = code[:2000]
    complexity = analyze_code_complexity(code)
    load_level = complexity['level']
    eeg = current_eeg_state()
    suggestion = get_ollama_suggestion(code, load_level, complexity, eeg['state'])
    return jsonify({
        'code':       code,
        'load_level': load_level,
        'complexity': complexity,
        'eeg':        eeg,
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

def eye_status():
    """Latest webcam blink data, or None values if the eye tracker is not running."""
    fresh = time.time() - eye_data.get('updated', 0) < EYE_STALE_S
    return {'blink_rate': eye_data.get('blink_rate') if fresh else None,
            'eye_load':   eye_data.get('eye_load') if fresh else None,
            'detector':   eye_data.get('detector') if fresh else None,
            'active':     fresh}

@app.route('/update_eye', methods=['POST'])
def update_eye():
    global eye_data
    d = request.get_json(silent=True) or {}
    try:
        br = float(d.get('blink_rate'))
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'message': 'blink_rate missing'}), 400
    eye_data = {'blink_rate': round(br, 1), 'eye_load': d.get('eye_load'),
                'detector': d.get('detector'), 'updated': time.time()}
    return jsonify({'status': 'ok'})

@app.route('/eye_status')
def eye_status_route():
    return jsonify(eye_status())

@app.route('/status')
def status():
    return jsonify({
        'code_load':   'radon code complexity',
        'brain_model': 'XGBoost, reading vs rest, trained on 9 programmers (LOSO AUC 0.86)',
        'brain_input': 'EEG playback of participant P05 (unseen)',
        'features':    196,
        'status':      'active'
    })

if __name__ == '__main__':
    print("="*50)
    print("  Cognitive Load Backend v2 Starting...")
    print("  Code load: radon complexity | Brain state: EEG playback (/eeg_state)")
    print("  Suggestion engine: Ollama (llama3.2) with rule-based fallback")
    print("  URL: http://localhost:5000")
    print("="*50)
    app.run(port=5000, debug=False)