import tkinter as tk
import random
from threading import Lock
from typing import List, Tuple, Dict
from Color import COLORS

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
    
    #type hints
    def __init__(self) -> None:
        #2차원 필드 생성. height x width
        self.field = [[0 for c in range(Tetris.FIELD_WIDTH)] for r in range(Tetris.FIELD_HEIGHT)]
        self.score = 0
        self.level = 0
        self.total_lines_eliminated = 0
        self.game_over = False
        self.move_lock = Lock()
        self.reset_mino()#미노 가져옴.
    
    def reset_mino(self)-> None: #미노를 하나 가져옵니다.
        self.mino = random.choice(Tetris.MINOS)[:]#랜덤으로 하나 가져옵니다.
        self.mino_color = random.randint(1, len(COLORS)-1)
        self.mino_offset = [-2, Tetris.FIELD_WIDTH//2] #위치 시작점.
        self.game_over = False

    def get_mino_coords(self)-> List[Tuple[int, int]]: #좌표를 얻습니다.
        return [(r+self.mino_offset[0], c + self.mino_offset[1]) for (r, c) in self.mino]
    
    def add_mino_queue(self, size = 1)-> None:
        #todo : impl this method
        pass

    def get_color(self, r, c):
        return self.mino_color if (r, c) in self.get_mino_coords() else self.field[r][c]

    def apply_mino(self):
            for (r, c) in self.get_mino_coords():
                self.field[r][c] = self.mino_color

            new_field = [r for r in self.field if any(tile == 0 for tile in r)]
            lines_eliminated = len(self.field)-len(new_field)
            self.total_lines_eliminated += lines_eliminated
            self.field = [[0]*Tetris.FIELD_WIDTH for x in range(lines_eliminated)] + new_field
            #self.score += Tetris.SCORE_PER_ELIMINATED_LINES[lines_eliminated] * (self.level + 1)
            self.level = self.total_lines_eliminated // 10
            self.reset_mino()

    def move(self, dr, dc):
        """
        move(0, -1 ) move (0, 1) 로 호출.
        """
        with self.move_lock:
            if self.game_over:
                return

            if all(self.is_cell_free(r + dr, c + dc) for (r, c) in self.get_mino_coords()):#놓을 수 있으면
                self.mino_offset = [self.mino_offset[0] + dr, self.mino_offset[1] + dc]
            elif dr == 1 and dc == 0:
                self.game_over = any(r < 0 for (r, c) in self.get_mino_coords())
                if not self.game_over:
                    self.apply_mino() #현재 미노를 필드에 추가.

    def rotate(self, dir):
        """
        dir = 0 반시계 방향 왼쪽으로 90도 회전
        dir = 1 시계 방향 오른쪽으로 90도 회전
        """
        with self.move_lock:
            if self.game_over:
                self.game_over_action()
                return
            
            row_list =[r for(r,c) in self.mino]
            col_list =[c for(r,c) in self.mino]
            size = max(max(col_list) - min(col_list), max(row_list)-min(row_list))
            rotated_mino = [(c, size-r) for (r, c) in self.tetromino]
            
            if dir == 0 :
                rotated_mino = [(size-c,r) for (r, c) in self.mino]
            elif dir == 1:
                rotated_mino = [(c, size-r) for (r, c) in self.mino]

            next_offset = self.mino_offset[:]
            mino_coord = [(r + next_offset[0], c + next_offset[1]) for (r,c) in rotated_mino]

            if all(self.is_cell_free(r, c) for (r, c) in mino_coord):
                    self.mino, self.mino_offset = rotated_mino, next_offset
                    print('qkfds')
        #todo : 충돌 구현
    
    def is_cell_free(self, r, c):#(r,c) 위치가 0 인지 체크.
        return r < Tetris.FIELD_HEIGHT and 0 <= c < Tetris.FIELD_WIDTH and (r < 0 or self.field[r][c] == 0)
            
    def game_over_action(self) -> None:
        #todo : imple this method
        print('end game')