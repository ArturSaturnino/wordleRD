import numpy as np # type: ignore
from typing import cast, List, Optional, Dict, Tuple

ALPHABET_SIZE = ord('z') - ord('a') + 1

class Utils:

    @staticmethod
    def toArray(word:str) -> np.ndarray[tuple[int, int], np.int32]:

        arr: np.ndarray[tuple[int, int], np.int32] = np.full(shape=(ALPHABET_SIZE, len(word)), fill_value = 0, dtype=np.int32)
        for i, c in enumerate(word):
            arr[ord(c) - ord('a'), i] = 1
        return arr


class State:
    _wordSize: int
    _upperBounds: np.ndarray[tuple[int], np.int32]
    _lowerBounds: np.ndarray[tuple[int], np.int32]
    _knownValues: np.ndarray[tuple[int, int], np.int32]
    _wrongValues: np.ndarray[tuple[int, int], np.int32]

    def __init__(self, wordSize:int):

        self._wordSize = wordSize

        self._upperBounds = np.full(shape=(ALPHABET_SIZE,), fill_value=wordSize, dtype=np.int32)
        self._lowerBounds = np.full(shape=(ALPHABET_SIZE,), fill_value=0, dtype=np.int32)
        self._knownValues = np.full(shape=(ALPHABET_SIZE, self._wordSize), fill_value= 0, dtype=np.int32)
        self._wrongValues = np.full(shape=(ALPHABET_SIZE, self._wordSize), fill_value= 0, dtype=np.int32)

    @property
    def lowerBounds(self) -> np.ndarray[tuple[int], np.int32]:
        return self._lowerBounds

    @property
    def upperBounds(self) -> np.ndarray[tuple[int], np.int32]:
        return self._upperBounds

    @property
    def knownValues(self) -> np.ndarray[tuple[int, int], np.int32]:
        return self._knownValues

    @property
    def wrongValues(self) -> np.ndarray[tuple[int, int], np.int32]:
        return self._wrongValues


class Engine:
    _secretArr: np.ndarray[tuple[int, int], np.int32]
    _secret: str
    _state: State
    _vocab: List[str]
    _vocabIdx: Dict[str, int]
    _vocabArray: np.ndarray[tuple[int, int, int], np.int32]
    _history: Optional[List[str]]
    _nTries: int

    def __init__(self, secret: str, vocab: List[str], record: bool=False):
        self._secretArr = Utils.toArray(secret)
        self._secret = secret
        self. _state = State(len(secret))
        self._vocab = vocab
        self._history = [] if record else None
        self._nTries = 0
        self._vocabIdx = {w: i for i, w in enumerate(self._vocab)}
        self._vocabArray = cast(np.ndarray[tuple[int, int, int], np.int32], np.array([Utils.toArray(w) for w in self._vocab]))

    def guess(self, guess: str, updateState: bool=True) -> bool:

        self._nTries += 1

        if self._history is not None:
            self._history.append(guess)

        if updateState and guess in self._vocabIdx.keys():

            guessArr = self._vocabArray[self._vocabIdx[guess], :]
            wordCountGuess = np.sum(guessArr, axis = -1)
            wordCountSecret = np.sum(self._secretArr, axis = -1)

            self._state._lowerBounds = cast(np.ndarray[tuple[int], np.int32], np.maximum(np.minimum(wordCountGuess, wordCountSecret), self._state.lowerBounds))
            self._state._upperBounds = cast(np.ndarray[tuple[int], np.int32], np.where(wordCountGuess > wordCountSecret, wordCountSecret, self._state.upperBounds))

            self._state._knownValues = (guessArr & self._secretArr) | self._state.knownValues
            self._state._wrongValues = (guessArr ^ self._secretArr) & guessArr | self._state.wrongValues

        return guess == self._secret

    def getState(self) -> State:
        return self._state

    def getFeasibleSet(self) -> np.ndarray[tuple[int], np.dtype[bool]]:

        letters = np.sum(self._vocabArray, -1)
        feasebleSet: np.ndarray[tuple[int], np.dtype[bool]] = np.all(np.less_equal(self._state.lowerBounds, letters), axis=-1)
        feasebleSet  &= np.all(np.less_equal(letters, self._state.upperBounds), axis = -1)
        feasebleSet &= np.all((self._state.knownValues & self._vocabArray) == self._state.knownValues, axis = (-1,-2))
        feasebleSet &= ~np.any(self._state.wrongValues & self._vocabArray, axis = (-1,-2))

        return feasebleSet

    def getFeasibleSetAverage(self) -> np.ndarray[tuple[int, int], np.float32]:
        return np.average(self._vocabArray[self.getFeasibleSet(), :], axis = 0)

    def getNTries(self) -> int:
        return self._nTries

    def getVocab(self) -> List[str]:
        return self._vocab

    def getVocabArray(self) -> np.ndarray[tuple[int, int, int], np.int32]:
        return self._vocabArray

    def getHistory(self) -> Optional[List[str]]:
        return self._history




