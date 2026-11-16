import json
import pathlib
import tempfile
import unittest
from .efficiency import build_stats, generation_counts


class BuildCostsTest(unittest.TestCase):
    def test_failed_retrieval_is_not_a_fast_generation(self):
        preds = [{'error': 'retrieval_timeout'},
                 {'error': None, 'latency': 1},
                 {'error': 'ReadTimeout', 'latency': 2}]
        stats = generation_counts(preds, 2)
        self.assertEqual(stats['n_generation_attempts'], 2)
        self.assertEqual(stats['n_retrieval_invalid'], 1)
        self.assertEqual(stats['n_generation_errors'], 1)
        self.assertEqual(stats['questions_per_second'], 1)
        self.assertEqual(stats['records_per_second'], 1.5)
        self.assertEqual(generation_counts(preds[:1], 0.01)['questions_per_second'], 0)

    def test_missing_is_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            stats = build_stats(pathlib.Path(d))
        self.assertIsNone(stats['build_qwen_calls'])
        self.assertIsNone(stats['builder_input_tokens'])
        self.assertIsNone(stats['build_wall_seconds'])

    def test_counts_only_build_and_keeps_partial_usage_visible(self):
        with tempfile.TemporaryDirectory() as d:
            rec = pathlib.Path(d)
            (rec / 'agent_meta.json').write_text(json.dumps({
                'start': '2026-10-22T01:00:00+08:00',
                'end': '2026-10-22T02:00:00+08:00'}))
            (rec / 'agent_usage.json').write_text(json.dumps({'unified': {
                'input_tokens': 120, 'cached_input_tokens': 100,
                'output_tokens': 10, 'output_incomplete': True}}))
            call = {'event': 'call', 'usage': {'prompt_tokens': 20, 'completion_tokens': 5}}
            events = [call, {'event': 'agent_start'}, call, call]
            (rec / 'gateway_build.jsonl').write_text('\n'.join(map(json.dumps, events)) + '\n{broken\n')
            stats = build_stats(rec)
        self.assertEqual(stats['build_wall_seconds'], 3600)
        self.assertEqual(stats['build_qwen_calls'], 2)
        self.assertEqual(stats['build_qwen_prompt_tokens'], 40)
        self.assertEqual(stats['build_qwen_completion_tokens'], 10)
        self.assertEqual(stats['builder_input_tokens'], 120)
        self.assertTrue(stats['builder_output_incomplete'])
        self.assertEqual(stats['build_gateway_malformed_lines'], 1)


if __name__ == '__main__':
    unittest.main()
