import unittest
from wordleRl.wordleEngine import Utils, ALPHABET_SIZE, Engine
import numpy as np

from wordleRl.wordleGame import PlayerCenter, Game


class TestGame(unittest.TestCase):

    def test_toArray(self):
        expected: np.ndarray = np.full(shape=(ALPHABET_SIZE, 2), fill_value = 0, dtype=np.int32)

        expected[0,0] = 1
        expected[1,1] = 1
        self.assertTrue(np.all(Utils.toArray("ab") == expected))

    def test_guess(self):

        engine = Engine("abc", ["aad", "aab", "bca", "abc"])
        end = engine.guess("abc", updateState=False)
        self.assertTrue(end)

        engine = Engine("abc", ["aad", "aab", "bca", "abc"])
        notEnd = engine.guess("aad", updateState=True)
        newState = engine.getState()
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

        self.assertEqual(engine.getNTries(), 1)

    def test_feasible(self):
        engine = Engine("abcd", ["abcd", "aacb", "aacd", "abce"])
        engine.guess("aacb", updateState=True)
        feasSet = engine.getFeasibleSet()
        self.assertFalse(np.any(feasSet ^ np.array([True, False, False, True])))

        averg = engine.getFeasibleSetAverage()
        expAverg = np.zeros(shape=(ALPHABET_SIZE, 4), dtype=np.float32)
        expAverg[0, 0] = 1
        expAverg[1, 1] = 1
        expAverg[2, 2] = 1
        expAverg[3, 3] = 0.5
        expAverg[4, 3] = 0.5

        np.testing.assert_allclose(averg, expAverg, 1e-6)

    def test_game(self):
        player = PlayerCenter()
        game = Game(player, vocab=["aaa", "aab", "bcd", "abc"], secret="abc", record=True)

        self.assertEqual(player.makeGuess(game.getEngine()), "aaa")

        game.play()
        self.assertListEqual(game.getEngine().getHistory(), ["aaa", "abc"])



if __name__ == '__main__':
    unittest.main()
