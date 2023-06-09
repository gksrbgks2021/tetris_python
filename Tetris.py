import tkinter as tk
import random
from threading import Lock
from typing import List, Tuple, Dict
from Color import COLORS
import numpy as np
import time
import pickle

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
        
        self.init_variable()
        self.init_setting()
        self.reset_mino()#미노 가져옴.
        
        #큐러닝 attribute
        #self.q_table = self.initialize_q_table()
        
    def init_variable(self):
        """
            싱글톤
        """
        self.move_lock = Lock()
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1
        self.num_actions = 3
        self.isbottom = False
    
    def show_field(self):
        for x in self.field:
            print(x)
            
    def init_setting(self):
        """
            게임 종료 후 필드를 초기화 합니다. 
        """
        self.field = [[0 for c in range(Tetris.FIELD_WIDTH)] for r in range(Tetris.FIELD_HEIGHT)]
        self.score = 0
        self.level = 0
        self.total_lines_eliminated = 0
        self.game_over = False   
        self.isbottom = False     
    
    def reset_mino(self)-> None: #미노를 하나 가져옵니다.
        self.mino_index = random.randint(0,6)
        self.mino =Tetris.MINOS[self.mino_index][:]#랜덤으로 하나 가져옵니다.
        self.mino_color = random.randint(1, len(COLORS)-1)
        self.mino_offset = [-2, Tetris.FIELD_WIDTH//2] #offset 시작점.(-2행, 4열)
        self.isbottom = False

    def get_mino_coords(self): #좌표를 얻습니다.
        return [(r+self.mino_offset[0], c + self.mino_offset[1]) for (r, c) in self.mino]
    
    def add_mino_queue(self, size = 1)-> None:
        #todo : impl this method
        pass

    def get_color(self, r, c):
        return self.mino_color if (r, c) in self.get_mino_coords() else self.field[r][c]

    def apply_mino(self) -> None:
            for (r, c) in self.get_mino_coords():
                self.field[r][c] = self.mino_color

            new_field = [r for r in self.field if any(tile == 0 for tile in r)]
            #제거 된 라인 수 
            lines_eliminated = len(self.field)-len(new_field)
            
            self.total_lines_eliminated += lines_eliminated
            self.field = [[0]*Tetris.FIELD_WIDTH for x in range(lines_eliminated)] + new_field
            self.score += lines_eliminated * 500
            self.level = self.total_lines_eliminated // 10
            self.reset_mino()

    def move(self, dr, dc) -> None:
        """
        call ex ) move(0, -1 ) move (0, 1) 
        """
        with self.move_lock:
            if self.game_over:
                return
            continue_flag = False

            if all(self.is_cell_free(r + dr, c + dc) for (r, c) in self.get_mino_coords()):#if not collision
                self.mino_offset = [self.mino_offset[0] + dr, self.mino_offset[1] + dc] #move offset
                continue_flag = True
            elif dr == 1 and dc == 0:
                #현재 좌표에서 행이 마이너스가 되면 게임 끝.
                self.game_over = any(r < 0 for (r, c) in self.get_mino_coords())
                if not self.game_over:
                    self.apply_mino() #현재 미노를 필드에 추가.
                    
            return continue_flag

    def get_mino_size(self) -> int:
        row_list , col_list = zip(*self.mino)
        return max(max(row_list)-min(row_list), max(col_list) - min(col_list))
        
    def rotate(self, dir):
        """
        dir = 0 반시계 방향 rotate left 90 d
        dir = 1 시계 방향 rotate right 90 d
        """
        with self.move_lock:
            if self.game_over:
                self.game_over_action()
                return
            
            size = self.get_mino_size()
            rotated_mino = [(c, size-r) for (r, c) in self.mino]

            if dir == 0 :
                rotated_mino = [(size-c,r) for (r, c) in self.mino]
            elif dir == 1:
                rotated_mino = [(c, size-r) for (r, c) in self.mino]

            next_offset = self.mino_offset[:]
            next_mino_coord = [(r + next_offset[0], c + next_offset[1]) for (r,c) in rotated_mino]
            
            #필드 밖일때 예외처리
            min_pos_x = min(c for (r, c) in next_mino_coord)
            max_pos_x = max(c for (r, c) in next_mino_coord)
            max_pos_y = max(r for (r, c) in next_mino_coord)
            
            next_offset[1] -= min(0, min_pos_x) # x 좌표 : -1 , -2 일때 예외처리
            next_offset[1] += min(0, Tetris.FIELD_WIDTH - (1 + max_pos_x)) # 필드 x 좌표 벗어난 만큼 뺄셈.
            next_offset[0] += min(0, Tetris.FIELD_HEIGHT - (1 + max_pos_y))# 필드 y 좌표 벗어난 만큼 뺄셈.

            next_mino_coord = [(r + self.mino_offset[0], c + self.mino_offset[1]) for (r,c) in rotated_mino]
            
            if all(self.is_cell_free(r, c) for (r, c) in next_mino_coord):
                self.mino, self.mino_offset = rotated_mino, next_offset
    
    def is_cell_free(self, r, c) -> bool :#(r,c) 위치가 0 인지 체크.
        return r < Tetris.FIELD_HEIGHT and 0 <= c < Tetris.FIELD_WIDTH and (r < 0 or self.field[r][c] == 0)
            
    def game_over_action(self) -> None:
        #todo : imple this method
        print('end game')
    def __get_mino_index__(self)->int:
        return self.mino_index