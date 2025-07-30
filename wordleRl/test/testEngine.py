import unittest
from wordleRl.wordleEngine import Utils, ALPHABET_SIZE, Engine
import numpy as np


class TestEngine(unittest.TestCase):

    def test_toArray(self):
        expected: np.ndarray = np.full(shape=(ALPHABET_SIZE, 2), fill_value = 0, dtype=np.int32)

        expected[0,0] = 1
        expected[1,1] = 1
        self.assertTrue(np.all(Utils.toArray("ab") == expected))

    def test_guess(self):

        engine = Engine("abc", ["aad", "aab", "bca", "abc"])
        end, _ = engine.guess("abc", updateState=False)
        self.assertTrue(end)

        engine = Engine("abc", ["aad", "aab", "bca", "abc"])
        notEnd, newState = engine.guess("aad", updateState=True)
        self.assertFalse(notEnd)

        self.assertTrue(newState.knownValues[0,0] == 1)
        self.assertTrue(newState.knownValues.sum() == 1)

        self.assertTrue(newState.wrongValues[0, 1] == 1)
        self.assertTrue(newState.wrongValues[3, 2] == 1)
        self.assertTrue(newState.wrongValues.sum() == 2)

        self.assertTrue(newState.lowerBonds[0] == 1)
        self.assertTrue(newState.lowerBonds.sum() == 1)

        self.assertTrue(newState.upperBonds[0] == 1)
        self.assertTrue(newState.upperBonds[3] == 0)
        self.assertTrue(newState.upperBonds.sum() == 3*ALPHABET_SIZE - 5)

    def test_feasable(self):
        engine = Engine("abcd", ["abcd", "aacb", "aacd", "abce"])
        end, _ = engine.guess("aacb", updateState=True)
        feasSet = engine.getFeasebleSet()
        self.assertFalse(np.any(feasSet ^ np.array([True, False, False, True])))





if __name__ == '__main__':
    unittest.main()
