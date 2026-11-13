import radon.complexity as rc
import radon.metrics as rm


def _parse_trimmed(code_text):
    """Code sent from the editor may be cut mid-function (the extension sends at most
    2000 characters). Drop lines from the end until radon can parse it, so we measure
    the complete part instead of failing."""
    lines = code_text.split('\n')
    for cut in range(len(lines), 0, -1):
        part = '\n'.join(lines[:cut])
        try:
            return rc.cc_visit(part), part
        except Exception:
            continue
    raise ValueError('no parsable code')


# Default = Intermediate profile (see personalized_threshold.py and thesis Table 5.1)
DEFAULT_T = {'high_cc': 4, 'medium_cc': 2, 'high_loops': 2, 'medium_loops': 1}


def _count_lines(code_text):
    # ignore comments and blank lines when counting loops and conditions
    lines = [l.split('#')[0] for l in code_text.split('\n')]
    lines = [l for l in lines if l.strip()]
    loops = sum(1 for l in lines if l.strip().startswith(('for ', 'while ')))
    conds = sum(1 for l in lines if l.strip().startswith(('if ', 'elif ')))
    funcs = sum(1 for l in lines if l.strip().startswith('def '))
    return loops, conds, funcs


def analyze_code_complexity(code_text, thresholds=None):
    t = thresholds or DEFAULT_T
    try:
        blocks, code_text = _parse_trimmed(code_text)

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
        except Exception:
            halstead_effort = 0

        loop_count, condition_count, function_count = _count_lines(code_text)

        if max_cc >= t['high_cc'] or loop_count >= t['high_loops'] or halstead_effort > 500:
            level = 'High'
        elif max_cc >= t['medium_cc'] or loop_count >= t['medium_loops'] or condition_count >= 2:
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

    except Exception:
        # Not parsable at all (e.g. not Python): fall back to simple line counts
        loop_count, condition_count, _ = _count_lines(code_text)

        if loop_count >= t['high_loops']:
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