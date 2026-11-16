import json,pathlib,subprocess,sys,tempfile,unittest


class ExitClassificationTest(unittest.TestCase):
    def test_signal_codes_are_checked_against_the_actual_deadline(self):
        cases=[(137,'00:50:00',1,True),(137,'01:01:00',0,False),
               (124,'01:00:00',0,False),(124,'00:50:00',1,True),
               (42,'00:50:00',1,True),(0,'00:50:00',0,False)]
        for code,end,crashes,suspect in cases:
            with self.subTest(code=code,end=end),tempfile.TemporaryDirectory() as folder:
                p=pathlib.Path(folder)
                (p/'stderr').write_text('[2026-10-22T00:00:00+08:00] dsh initial (left=3600s)\n'
                    f'[2026-10-22T{end}+08:00] dsh initial exit={code}\n')
                (p/'trace').write_text('')
                (p/'meta').write_text(json.dumps({'deadline':'2026-10-22T01:00:00+08:00','exit_code':0}))
                subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('agent_segments.py')),
                    str(p/'stderr'),str(p/'trace'),str(p/'meta')],check=True,capture_output=True)
                result=json.loads((p/'meta').read_text())
                self.assertEqual(result['session_stats']['n_crashed_segments'],crashes)
                self.assertEqual(result['session_stats']['platform_suspect'],suspect)
                self.assertEqual(result['exit_code'],0)  # The actual wrapper status remains original evidence.


if __name__=='__main__':unittest.main()
