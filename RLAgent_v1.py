from threading import Lock
from Tetris import Tetris
import tkinter as tk
from Color import COLORS
import numpy as np
import copy
import random
import math
from typing import List, Tuple, Dict

class RLAgent_v1(Tetris):

    #미노 회전할수 있는 상태. 
    #O,L,J, Z, T, S, I
    move_cnt = [1,4,4,2,4,2,2]
    
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.weights = [-1, -1, -1, -30]
        self.explore_change = 0.5
        self.alpha = 0.01
        self.gamma = 0.9
        print(self.num_actions)

    def move_simul(self, dr, dc,field, mino,offset, stop_flag) : # need 6 param
        """
        move(0, -1 ) move (0, 1) 로 호출.
        """
        with self.move_lock:
            if stop_flag:
                return mino, offset

            if all(self.is_tempcell_free(field,r + dr, c + dc) for (r, c) in self.get_mino_temp_coords(mino, offset)):#놓을 수 있으면
                offset = [offset[0] + dr, offset[1] + dc]
            elif dr == 1 and dc == 0: # 아래로 내려갈떄
                stop_flag = any(r < 0 for (r, c) in self.get_mino_temp_coords(mino, offset))
                if not self.game_over:
                    self.apply_mino_simul(field,mino, offset) #현재 미노를 필드에 추가.
                    pass
            return mino, offset, stop_flag
        
    def is_tempcell_free(self,map, r, c) -> bool :#(r,c) 위치가 0 인지 체크.
        return r < Tetris.FIELD_HEIGHT and 0 <= c < Tetris.FIELD_WIDTH and (r < 0 or map[r][c] == 0)

    def apply_mino_simul(self, field,mino, offset) -> None:#시뮬레이션 적용.
            for (r, c) in self.get_mino_temp_coords(mino, offset):
                field[r][c] = self.mino_color
            #하나라도 0 이면 카운트
            new_field = [r for r in field if any(tile == 0 for tile in r)]
            #제거 된 라인 수 
            lines_eliminated = len(field)-len(new_field)
            return lines_eliminated

    def get_mino_temp_coords(self,mino,offset): #좌표를 얻습니다.
        return [(r+offset[0], c + offset[1]) for (r, c) in mino]
    
    def rotate_simul(self, map, mino, offset):
        with self.move_lock:
            if self.game_over:
                self.game_over_action()
                return
            
            size = self.get_mino_size()
            rotated_mino = [(c, size-r) for (r, c) in mino]

            next_offset = offset[:]
            next_mino_coord = [(r + next_offset[0], c + next_offset[1]) for (r,c) in rotated_mino]
            
            #필드 밖일때 예외처리
            min_pos_x = min(c for (r, c) in next_mino_coord)
            max_pos_x = max(c for (r, c) in next_mino_coord)
            max_pos_y = max(r for (r, c) in next_mino_coord)
            
            next_offset[1] -= min(0, min_pos_x) # x 좌표 : -1 , -2 일때 예외처리
            next_offset[1] += min(0, Tetris.FIELD_WIDTH - (1 + max_pos_x)) # 필드 x 좌표 벗어난 만큼 뺄셈.
            next_offset[0] += min(0, Tetris.FIELD_HEIGHT - (1 + max_pos_y))# 필드 y 좌표 벗어난 만큼 뺄셈.

            next_mino_coord = [(r + next_offset[0], c + next_offset[1]) for (r,c) in rotated_mino]
            
            if all(self.is_tempcell_free(map, r, c) for (r, c) in next_mino_coord):
                mino = rotated_mino
                offset = next_offset
            return mino , offset
    
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
        for j in range(0, Tetris.FIELD_WIDTH):
            for i in range(0, Tetris.FIELD_HEIGHT):
                if int(map[i][j]) > 0:
                    heights[j] = Tetris.FIELD_HEIGHT - i
                    break

        # 높이 차 배열
        for i in range(0, len(diffs)):
            diffs[i] = heights[i + 1] - heights[i]

        # Calculate the maximum height
        max_height = max(heights)

        # 위에서 아래로 빈 셀의 개수(꼭대기는 1 인 셀)을 센다.
        for i in range(0, Tetris.FIELD_HEIGHT):
            occupied = 0 
            for j in range(0, Tetris.FIELD_WIDTH):  # Scan from top to bottom
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

    def simulate_map(self,map, mino, move):
        # 'move = (rotate, sideways)
        #  rot = [0:3] 미노 회전 횟수
        # sideways = [-9:9] 현재 위치에서 수평 이동
        # 전체 라인을 제거 다음 보드 상태와 지워진 라인 수를 반환

        rot = move[0]
        sideways = move[1]
        test_lines_removed = 0
        reference_height = self.get_state(map)[0]
        stop_flag = False

        if mino is None:
            return None
        
        #바꿀 offset
        offset = self.mino_offset

        # rotate
        for _ in range(rot):
            mino , offset = self.rotate_simul(map,mino,offset)
        
        # 양옆으로 움직임
        mino , offset, stop_flag =self.move_simul(0, sideways,map, mino, offset,stop_flag)

        # 충돌할떄까지 밑으로 내림
        while not stop_flag and all(self.is_tempcell_free(map,r + 1, c) for (r, c) in self.get_mino_temp_coords(mino, offset)):
            mino , offset, stop_flag = self.move_simul(1, 0, map, mino, offset, stop_flag)
        
        #보상으로 5*(제거된 라인 수)^2 - (이번 높이들의 max- prev 높이 max)
        #라인을 많이 제거하고, 높이 증가를 최소화하기위해 보상 설정

        height_sum, diff_sum, max_height, holes, height_std = self.get_state(map)
        one_step_reward = 5 * (test_lines_removed * test_lines_removed) - (height_sum - reference_height)
        return map, one_step_reward
    
    def find_best_move(self,map, mino, weights, explore_change):
        move_list = []
        score_list = []
        #이동 가능한 횟수
        for rot in range(0, RLAgent_v1.move_cnt[self.mino_index]):
            for sideways in range(-5, 6):
                move = [rot, sideways]#(회전 수 , 양옆으로 이동 수 ) 하나 만든다.
                field_copy = copy.deepcopy(map)
                mino_copy = copy.deepcopy(mino)
                test_map = self.simulate_map(field_copy, mino_copy, move)
                if test_map is not None:
                    move_list.append(move)#move 리스트 추가. 
                    test_score = self.get_expected_score(test_map[0], weights)
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
    def gradient_descent(self,weights):
        #move를 찾는다.
        move = self.find_best_move(self.field, self.mino, weights, self.__get_explore_change__())
        current_params = self.get_state(self.field)
        test_field = copy.deepcopy(self.field)
        test_mino = copy.deepcopy(self.mino)
        test_field = self.simulate_map(test_field, test_mino, move)

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
    
    def __get_explore_change__(self):
        return self.explore_change