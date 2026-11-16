"""Validate terminal build status without trusting a wrapper's masked exit code."""
import json
import sys
from datetime import datetime


def terminal_build_ok(meta):
    segments = meta.get('session_stats', {}).get('segments', [])
    if not segments:
        return False
    final = segments[-1]

    def normal(code, end):
        if code == 0:
            return True
        if code not in (124, 137) or not end or not meta.get('deadline'):
            return False
        try:
            return (datetime.fromisoformat(end).timestamp() >=
                    datetime.fromisoformat(meta['deadline']).timestamp() - 2)
        except (TypeError, ValueError):
            return False

    return normal(meta.get('exit_code'), meta.get('end')) and normal(final.get('exit'), final.get('end'))


if __name__ == '__main__':
    with open(sys.argv[1], encoding='utf-8') as stream:
        ok = terminal_build_ok(json.load(stream))
    print('BUILD_TERMINAL_STATUS', 'accepted' if ok else 'abnormal_or_incomplete')
    sys.exit(0 if ok else 1)
