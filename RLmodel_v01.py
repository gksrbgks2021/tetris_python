from threading import Lock
from Tetris import Tetris
import tkinter as tk
from Color import COLORS
import numpy as np
import copy
import random
import math
from typing import List, Tuple, Dict

class RLmodel_v01(Tetris):

    #미노 회전할수 있는 상태. 
    #O,L,J, Z, T, S, I
    move_cnt = [1,4,4,2,4,2,2]
    
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.weights = [-1, -1, -1, -30]
        self.alpha = 0.01
        self.gamma = 0.9
        print(self.num_actions)

    def move(self, dr, dc) -> None:
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
    
    def get_state(self, map):
        """
            heights : max(높이)
            diffs : heights[i+1] - heights[i] 차이 배열
            holes = 비어 있는 개수 
            max_height = 높이의 최댓값.
            diff_sum = diff 절댓값들의 합
            높이의 표준 편차. 
        """
        heights = [0]*Tetris.FIELD_WIDTH
        diffs = [0]*(Tetris.FIELD_WIDTH-1)
        holes = 0
        diff_sum = 0

        # 각 열의 꼭대기 인덱스 가져온다.
        for i in range(0, Tetris.FIELD_WIDTH):
            for j in range(0, Tetris.FIELD_HEIGHT):
                if int(map[i][j]) > 0:
                    heights[i] = Tetris.FIELD_HEIGHT - j
                    break

        # 높이 차 배열
        for i in range(0, len(diffs)):
            diffs[i] = heights[i + 1] - heights[i]

        # Calculate the maximum height
        max_height = max(heights)

        # 위에서 아래로 빈 셀의 개수(꼭대기는 1 인 셀)을 센다.
        for i in range(0, Tetris.FIELD_WIDTH):
            occupied = 0 
            for j in range(0, Tetris.FIELD_HEIGHT):  # Scan from top to bottom
                if int(map[i][j]) > 0:
                    occupied = 1  # If a block is found, set the 'Occupied' flag to 1
                if int(map[i][j]) == 0 and occupied == 1:
                    holes += 1  # If a hole is found, add one to the count

        height_sum = sum(heights)
        for x in diffs:
            diff_sum += abs(x)
        
        return height_sum, diff_sum, max_height, holes, np.std(heights)
    
    def get_expected_score(self,map, weights):
        # 가중치와, 이전에 제거된 라인 수가 주어졌을 때, 점수 계산해 리턴.
        state_values = self.get_state(map)
        expected_score = sum(weight * value for weight, value in zip(weights, state_values))
        return float(expected_score)

    def transition(self,map, mino, move):
        # 'move = (rotate, sideways)
        #  rot = [0:3] 미노 회전 횟수
        # sideways = [-9:9] 현재 위치에서 수평 이동
        # 전체 라인을 제거 다음 보드 상태와 지워진 라인 수를 반환

        rot = move[0]
        sideways = move[1]
        test_lines_removed = 0
        reference_height = self.get_state(map)[0]
        if mino is None:
            return None
        
        # rotate
        for _ in range(rot):
            self.rotate(1)
        
        # Move the mino sideways
        self.move(0, sideways)

        # Move the mino down until it collides
        while all(self.is_cell_free(r + 1, c) for (r, c) in self.get_mino_coords()):
            self.move(1, 0)
        
        #보상으로 5*(제거된 라인 수)^2 - (이번 높이들의 max- prev 높이 max)
        #라인을 많이 제거하고, 높이 증가를 최소화하기위해 보상을 이렇게 설정하였음

        height_sum, diff_sum, max_height, holes, height_std = self.get_state(map)
        one_step_reward = 5 * (test_lines_removed * test_lines_removed) - (height_sum - reference_height)
        return map, one_step_reward
    
    def find_best_move(self,board, piece, weights, explore_change):
        move_list = []
        score_list = []
        #이동 가능한 횟수
        for rot in range(0, len(RLmodel_v01.move_cnt[self.mino_index])):
            for sideways in range(-5, 6):
                move = [rot, sideways]
                field_copy = copy.deepcopy(board)
                mino_copy = copy.deepcopy(piece)
                test_board = self.transition(field_copy, mino_copy, move)
                if test_board is not None:
                    move_list.append(move)
                    test_score = self.get_expected_score(test_board[0], weights)
                    score_list.append(test_score)
        best_score = max(score_list)
        best_move = move_list[score_list.index(best_score)]

        #확률적으로 이동.
        if random.random() < explore_change:
            move = move_list[random.randint(0, len(move_list) - 1)]
        else:
            move = best_move
        return move
    
    #경사 하강법
    def gradient_descent(self,field, mino, weights, explore_change):

        move = self.find_best_move(field, mino, weights, explore_change)
        current_params = self.get_state(field)
        test_field = copy.deepcopy(field)
        test_mino = copy.deepcopy(mino)
        test_field = self.transition(test_field, test_mino, move)

        if test_field is not None:
            new_params = self.get_state(test_field[0])
            one_step_reward = test_field[1]
        
        #경사하강법 학습
        for i in range(0, len(weights)):
            weights[i] = weights[i] + self.alpha * weights[i] * (
                one_step_reward - current_params[i] + self.gamma * new_params[i])
        regularization_term = abs(sum(weights))
        #가중치를 정규화
        for i in range(0, len(weights)):
            weights[i] = 100 * weights[i] / regularization_term
            weights[i] = math.floor(1e4 * weights[i]) / 1e4  # Rounds the weights
        return move, weights