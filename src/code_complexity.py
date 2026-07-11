import radon.complexity as rc
import radon.metrics as rm

def analyze_code_complexity(code_text):
    try:
        blocks = rc.cc_visit(code_text)
        
        if blocks:
            scores = [block.complexity for block in blocks]
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

        lines = code_text.split('\n')
        loop_count      = sum(1 for l in lines if 'for ' in l or 'while ' in l)
        condition_count = sum(1 for l in lines if 'if ' in l or 'elif ' in l)
        function_count  = sum(1 for l in lines if l.strip().startswith('def '))

        if max_cc >= 4 or loop_count >= 2 or halstead_effort > 500:
            level = 'High'
        elif max_cc >= 2 or loop_count >= 1 or condition_count >= 2:
            level = 'Medium'
        else:
            level = 'Low'

        return {
            'level': level,
            'cyclomatic_complexity': round(avg_cc, 2),
            'max_complexity': max_cc,
            'loop_count': loop_count,
            'condition_count': condition_count,
            'function_count': function_count,
            'halstead_effort': round(halstead_effort, 2)
        }

    except Exception as e:
        lines = code_text.split('\n')
        loop_count      = sum(1 for l in lines if 'for ' in l or 'while ' in l)
        condition_count = sum(1 for l in lines if 'if ' in l)
        
        if loop_count >= 2:
            level = 'High'
        elif loop_count >= 1 or condition_count >= 1:
            level = 'Medium'
        else:
            level = 'Low'

        return {
            'level': level,
            'cyclomatic_complexity': 1,
            'max_complexity': 1,
            'loop_count': loop_count,
            'condition_count': condition_count,
            'function_count': 0,
            'halstead_effort': 0
        }


def get_stew_segment_for_complexity(level, data_T, class_labels, scaler, model):
    pass