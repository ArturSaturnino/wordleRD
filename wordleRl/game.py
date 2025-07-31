from abc import ABC, abstractmethod
import random
from typing import Optional, List
import numpy as np # type: ignore

import wordleRl.engine as wordleEngine


class Player(ABC):

    @abstractmethod
    def makeGuess(self, engine:wordleEngine.Engine) -> str:
        ...


class PlayerCenter(Player):

    def __init__(self):
        ...

    def makeGuess(self, engine:wordleEngine.Engine) -> str:
        target = engine.getFeasibleSetAverage()

        scores = np.sum(engine.getVocabArray() * target, axis=(-1,-2))
        idx = np.argmax(scores)

        return engine.getVocab()[idx]


class Game:

    _engine: wordleEngine.Engine
    _payer: Player
    _maxRounds: int

    def __init__(self,  player:Player, vocab:List[str], secret:Optional[str]=None, record:bool=False, maxRounds:int=100):
        self._engine = wordleEngine.Engine(secret if secret is not None else random.sample(vocab, 1)[0],  vocab, record)
        self._maxRounds = maxRounds
        self._payer = player

    def play(self) -> bool:
        end = False
        while (not end) and self._engine.getNTries() < self._maxRounds:
            end = self._engine.guess(self._payer.makeGuess(self._engine))
        return end

    def getEngine(self) -> wordleEngine.Engine:
        return self._engine

"""
vocab = []
with open("resources/words.txt", "r") as f:
    vocab = [w.strip('\n') for w in f.readlines()]

game = Game(PlayerCenter(), vocab, record=True, secret="assay")
game.play()
print(game.getEngine().getHistory())
"""