import json
import os

PROFILE_FILE = 'data/user_profile.json'

DEFAULT_THRESHOLDS = {
    'Beginner':     {'high_cc': 3, 'medium_cc': 2, 'high_loops': 1, 'medium_loops': 1},
    'Intermediate': {'high_cc': 4, 'medium_cc': 2, 'high_loops': 2, 'medium_loops': 1},
    'Expert':       {'high_cc': 6, 'medium_cc': 3, 'high_loops': 3, 'medium_loops': 2},
}

def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, 'r') as f:
            return json.load(f)
    return {'level': 'Intermediate', 'thresholds': DEFAULT_THRESHOLDS['Intermediate']}

def save_profile(level):
    profile = {
        'level':      level,
        'thresholds': DEFAULT_THRESHOLDS[level]
    }
    os.makedirs('data', exist_ok=True)
    with open(PROFILE_FILE, 'w') as f:
        json.dump(profile, f)
    print(f"Profile saved: {level}")
    return profile

def get_thresholds():
    profile = load_profile()
    return profile['thresholds']

def analyze_with_profile(code_text):
    import radon.complexity as rc
    import radon.metrics as rm

    thresholds = get_thresholds()
    profile    = load_profile()

    try:
        blocks = rc.cc_visit(code_text)
        if blocks:
            scores = [b.complexity for b in blocks]
            avg_cc = sum(scores) / len(scores)
            max_cc = max(scores)
        else:
            avg_cc = 1
            max_cc = 1

        try:
            h = rm.h_visit(code_text)
            halstead_effort = h[0].effort if h else 0
        except:
            halstead_effort = 0

        lines           = code_text.split('\n')
        loop_count      = sum(1 for l in lines if 'for ' in l or 'while ' in l)
        condition_count = sum(1 for l in lines if 'if ' in l or 'elif ' in l)
        function_count  = sum(1 for l in lines if l.strip().startswith('def '))

        if (max_cc >= thresholds['high_cc'] or
                loop_count >= thresholds['high_loops'] or
                halstead_effort > 500):
            level = 'High'
        elif (max_cc >= thresholds['medium_cc'] or
                loop_count >= thresholds['medium_loops'] or
                condition_count >= 2):
            level = 'Medium'
        else:
            level = 'Low'

        return {
            'level':                  level,
            'cyclomatic_complexity':  round(avg_cc, 2),
            'max_complexity':         max_cc,
            'loop_count':             loop_count,
            'condition_count':        condition_count,
            'function_count':         function_count,
            'halstead_effort':        round(halstead_effort, 2),
            'user_level':             profile['level'],
            'thresholds_used':        thresholds
        }

    except Exception:
        lines           = code_text.split('\n')
        loop_count      = sum(1 for l in lines if 'for ' in l or 'while ' in l)
        condition_count = sum(1 for l in lines if 'if ' in l)

        if loop_count >= thresholds['high_loops']:
            level = 'High'
        elif loop_count >= 1 or condition_count >= 1:
            level = 'Medium'
        else:
            level = 'Low'

        return {
            'level':                 level,
            'cyclomatic_complexity': 1,
            'max_complexity':        1,
            'loop_count':            loop_count,
            'condition_count':       condition_count,
            'function_count':        0,
            'halstead_effort':       0,
            'user_level':            profile['level'],
            'thresholds_used':       thresholds
        }


if __name__ == '__main__':
    print("Personalized Threshold Test")
    print("="*50)

    test_code = "def bubble_sort(arr):\n    for i in range(n):\n        for j in range(n-i-1):\n            if arr[j]>arr[j+1]:\n                arr[j],arr[j+1]=arr[j+1],arr[j]"

    for level in ['Beginner', 'Intermediate', 'Expert']:
        save_profile(level)
        result = analyze_with_profile(test_code)
        print(f"\n{level}: bubble_sort → {result['level']}")
        print(f"  CC threshold: {result['thresholds_used']['high_cc']}")
        print(f"  Loop threshold: {result['thresholds_used']['high_loops']}")