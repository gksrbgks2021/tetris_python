import tkinter as tk
import random
from threading import Lock

class Tetris():
    FIELD_HEIGHT = 20
    FIELD_WIDTH = 10

    MINOS = [
        [(0, 0), (0, 1), (1, 0), (1,1)], # O
        [(0, 0), (0, 1), (1, 0), (2,0)], # L
        [(0, 1), (1, 1), (2, 1), (2,0)], # J 
        [(0, 1), (1, 0), (1, 1), (2,0)], # Z
        [(0, 1), (1, 0), (1, 1), (2,1)], # T
        [(0, 0), (1, 0), (1, 1), (2,1)], # S
        [(0, 1), (1, 1), (2, 1), (3,1)], # I
    ]
    
    def __init__(self) -> None:
        #2차원 필드 생성. height x width
        self.field = [[0 for c in range(Tetris.FIELD_WIDTH)] for r in range(Tetris.FIELD_HEIGHT)]
        self.score = 0 
        self.level = 0
        self.game_over = False
        self.move_lock = Lock()
        self.reset_mino()
    
    def reset_mino(self)-> None:
        self.mino = random.choice(Tetris.MINOS)[:]#랜덤으로 하나 가져옵니다.
        self.mino_color = random.randint
        
    def add_mino_queue(self, size = 1)-> None:
        pass
        
