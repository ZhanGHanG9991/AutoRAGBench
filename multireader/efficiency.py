"""Read aggregate build costs without exporting trace text or gateway outputs."""
import datetime
import json


def generation_counts(predictions, elapsed):
    attempted = [p for p in predictions if 'latency' in p]
    return {
        'n_generation_attempts': len(attempted),
        'n_retrieval_invalid': len(predictions) - len(attempted),
        'n_generation_errors': sum(bool(p.get('error')) for p in attempted),
        'questions_per_second': len(attempted) / elapsed,
        'records_per_second': len(predictions) / elapsed,
    }


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def build_stats(rec):
    meta = load(rec / 'agent_meta.json')
    usage = load(rec / 'agent_usage.json').get('unified', {})
    freeze = load(rec / 'freeze.json')
    segments = meta.get('session_stats', {}).get('segments', [])
    seconds = None
    if meta.get('start') and meta.get('end'):
        seconds = (datetime.datetime.fromisoformat(meta['end']) -
                   datetime.datetime.fromisoformat(meta['start'])).total_seconds()
    gateway = rec / 'gateway_build.jsonl'
    calls = prompt = completion = bad_lines = 0
    started = False
    if gateway.exists():
        with gateway.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    bad_lines += 1
                    continue
                if event.get('event') == 'agent_start':
                    started = True
                if started and event.get('event') == 'call':
                    calls += 1
                    u = event.get('usage') or {}
                    prompt += u.get('prompt_tokens', 0) or 0
                    completion += u.get('completion_tokens', 0) or 0
    return {
        'builder_model': meta.get('model'), 'builder_harness': meta.get('harness'),
        'builder_harness_version': meta.get('harness_version'),
        'build_wall_seconds': seconds,
        'build_active_fraction': meta.get('session_stats', {}).get('active_fraction'),
        'build_reprompts': meta.get('session_stats', {}).get('n_reprompts'),
        'build_crashed_segments': meta.get('session_stats', {}).get('n_crashed_segments'),
        'build_platform_suspect': meta.get('session_stats', {}).get('platform_suspect'),
        'build_agent_exit_code': meta.get('exit_code'),
        'build_final_segment_exit': segments[-1].get('exit') if segments else None,
        'build_deadline': meta.get('deadline'),
        'builder_input_tokens': usage.get('input_tokens'),
        'builder_cached_input_tokens': usage.get('cached_input_tokens'),
        'builder_cache_write_tokens': usage.get('cache_write_tokens'),
        'builder_output_tokens': usage.get('output_tokens'),
        'builder_output_incomplete': usage.get('output_incomplete'),
        'build_qwen_calls': calls if started else None,
        'build_qwen_prompt_tokens': prompt if started else None,
        'build_qwen_completion_tokens': completion if started else None,
        'build_gateway_malformed_lines': bad_lines if gateway.exists() else None,
        'build_gateway_agent_start_seen': started if gateway.exists() else None,
        'workspace_bytes': freeze.get('workspace_bytes'),
    }
