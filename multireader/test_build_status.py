import unittest
from .build_status import terminal_build_ok


class BuildStatusTest(unittest.TestCase):
    def test_masked_and_unmasked_early_failures(self):
        for wrapper, child, end, expected in [
            (0, 0, '00:50:00', True), (0, 137, '00:50:00', False),
            (124, 124, '00:50:00', False), (124, 124, '01:00:00', True),
            (137, 137, '01:01:00', True), (42, 0, '01:01:00', False),
            (0, None, '01:01:00', False), (0, 42, '00:50:00', False),
        ]:
            with self.subTest(wrapper=wrapper, child=child, end=end):
                timestamp = '2026-10-22T' + end + '+08:00'
                meta = {'exit_code': wrapper, 'end': timestamp,
                        'deadline': '2026-10-22T01:00:00+08:00',
                        'session_stats': {'segments': [{'exit': child, 'end': timestamp}]}}
                self.assertEqual(terminal_build_ok(meta), expected)
        self.assertFalse(terminal_build_ok({'exit_code': 0}))


if __name__ == '__main__':
    unittest.main()
