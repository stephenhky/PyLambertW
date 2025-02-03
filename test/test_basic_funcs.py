
import unittest

import pylambertw.tukeyhutils as tutils


class TestBasic(unittest.TestCase):
    def test_tukeyh(self):
        assert tutils.tukeyh(0., 1.) == 0.


if __name__ == '__main__':
    unittest.main()
