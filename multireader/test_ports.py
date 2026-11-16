import unittest
from .ports import candidates

class PortTests(unittest.TestCase):
    def test_default_ephemeral_range_is_excluded(self):
        for job in range(1000):
            xs=candidates(job,(32768,60999))
            self.assertEqual(len(xs),200);self.assertEqual(len(set(xs)),200)
            self.assertTrue(all(b+59<32768 for b in xs))
    def test_nonstandard_kernel_range(self):
        xs=candidates(682690,(15000,25000))
        self.assertTrue(xs);self.assertTrue(all(b+59<15000 or b>25000 for b in xs))
    def test_no_safe_candidates(self):self.assertEqual(candidates(1,(1024,65535)),[])
    def test_current_workers_have_distinct_initial_blocks_per_node(self):
        for jobs in [[682688,682689,682690,687872,687873,687874,687877],[687875,687876,687878,687879,687880,689647,689648]]:
            xs=[candidates(j,(32768,60999))[0] for j in jobs];self.assertEqual(len(xs),len(set(xs)))

if __name__=='__main__':unittest.main()
