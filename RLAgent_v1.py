from threading import Lock
from Tetris import Tetris
import tkinter as tk
from Color import COLORS
import numpy as np
import copy
import random
import math
from typing import List, Tuple, Dict


ALPHA = 0.01
GAMMA = 0.9 #discount factor

class RLAgent_v1(Tetris):

    #미노 회전할수 있는 상태.
    #O,L,J, Z, T, S, I
    move_cnt = [1,4,4,2,4,2,2]
    
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.weights = []
        self.explore_change = 0.5 # 탐색할지 말지 확률
        self.decay_rate = 0.99
        self.alpha = 0.01
        self.gamma = 0.9
        self.evaluation_interval = 100
        self.evaluation_scores = []

    def move_simul(self, dr, dc,field, mino,offset, stop_flag) : # need 6 param
        """
        move_simul(0, -1 ) move_simul (0, 1) 로 호출.
        
        알고리즘
            1. 놓을 수 있으면, offset만 update 하고 리턴.
            2. 놓을수 없으면 stop_flag= true
        """
        with self.move_lock:
            if stop_flag:
                return mino, offset

            if all(self.is_tempcell_free(field,r + dr, c + dc) for (r, c) in self.get_mino_temp_coords(mino, offset)):
                offset = [offset[0] + dr, offset[1] + dc]
            else : 
                stop_flag = True
            return mino, offset, stop_flag
        
    def is_tempcell_free(self,map, r, c) -> bool :#(r,c) 위치가 0 인지 체크.
        return r < Tetris.FIELD_HEIGHT and 0 <= c < Tetris.FIELD_WIDTH and (r < 0 or map[r][c] == 0)

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
            heights : 각 열에서 max(높이) 들의 합
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
        
        #print(f'{Tetris.FIELD_WIDTH} , {Tetris.FIELD_HEIGHT} , {np.shape(map)}')
        # 각 열에 쌓인 높이들을 구한다.
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
        for j in range(0, Tetris.FIELD_WIDTH):
            occupied = 0 
            for i in range(0, Tetris.FIELD_HEIGHT):  # 위에서 아래로 스캔
                if int(map[i][j]) > 0:
                    occupied = 1  # If a block is found, set the 'Occupied' flag to 1
                if int(map[i][j]) == 0 and occupied == 1:
                    holes += 1  # If a hole is found, add one to the count

        height_sum = sum(heights)
        for x in diffs:
            diff_sum += abs(x)
        
        return height_sum, diff_sum, max_height, holes
    
    def get_expected_score(self,f, weights):
        # 가중치와, 이전에 제거된 라인 수가 주어졌을 때, 점수 계산해 리턴.
        state_values = self.get_state(f)
        expected_score = 0.0
        
        for w, v in zip(weights, state_values):
            expected_score += w * v
        
        return float(expected_score)

    def simulate_map(self,map, mino, move):
        """
        parameter 설명
            move = (rotate, sideways)
            rot = [0:3] 미노 회전 횟수
            sideways = [-9:9] 현재 위치에서 수평 이동할 열.
        
        return 값
            전체 라인을 제거 다음 보드 상태와 지워진 라인 수를 반환
        """

        rot = move[0]
        sideways = move[1]
        test_lines_removed = 0
        prev_state = self.get_state(map)
        stop_flag = False

        if mino is None:
            return None
        
        #바꿀 offset
        offset = copy.deepcopy(self.mino_offset)

        # rotate
        for _ in range(rot):
            mino , offset = self.rotate_simul(map,mino,offset)
        
        # 양옆으로 움직임
        mino , offset, stop_flag =self.move_simul(0, sideways,map, mino, offset,stop_flag)

        # 충돌할떄까지 밑으로 내림
        while not stop_flag and all(self.is_tempcell_free(map,r + 1, c) for (r, c) in self.get_mino_temp_coords(mino, offset)):
            mino , offset, stop_flag = self.move_simul(1, 0, map, mino, offset, stop_flag)
        map, test_lines_removed = self.apply_mino_simul(map,mino,offset)
        
        #보상으로 5*(제거된 라인 수)^2 - (이번 높이들의 max- prev 높이 max)
        #라인을 많이 제거하고, 높이 증가를 최소화하기위해 보상 설정

        #height_sum, diff_sum, max_height, holes, height_std = self.get_state(map)
        current_state = self.get_state(map)
        
        one_step_reward = 5 * (test_lines_removed * test_lines_removed) - 0.5 * (current_state[0] - prev_state[0]) - (current_state[2] - prev_state[2])
        #print(f'height_sum : {height_sum} and one_step_reward {one_step_reward}')
        
        return map, one_step_reward
    
    def apply_mino_simul(self, field, mino, offset) -> int:
        for (r, c) in self.get_mino_temp_coords(mino, offset):
            field[r][c] = self.mino_color

        new_field = [r for r in field if any(tile == 0 for tile in r)]
        lines_eliminated = len(field) - len(new_field)
        for _ in range(lines_eliminated):
            new_field.insert(0, [0]*Tetris.FIELD_WIDTH)
        return new_field, lines_eliminated
    
    def find_best_move(self,map, mino, weights, explore_change):
        move_list = []
        #action_value = 각 action 취했을때 reward 기댓값
        #2d 배열을 만든다.
        move_cnt = RLAgent_v1.move_cnt[self.__get_mino_index__()]
        score_list = []
        #이동 가능한 횟수
        for rot in range(move_cnt):
            for col_dif in range(-5, 5):
                move = [rot, col_dif]#(회전 수 , 양옆으로 이동 수 ) 하나 만든다.
                
                field_copy = copy.deepcopy(map)
                mino_copy = copy.deepcopy(mino)
                test_map, reward = self.simulate_map(field_copy, mino_copy, move)
                #print(f'move {move} 일때 보상 {reward}')
                if test_map is not None:
                    move_list.append(move)#move 리스트 추가. 
                    test_score = self.get_expected_score(test_map, weights)
                    score_list.append(test_score)
                    
        score_list = np.array(score_list)
        best_score = max(score_list)
        #print(f'best_score : {best_score}')
        idx_max = score_list.argmax()
        best_move = move_list[idx_max] #argmax

        #확률적으로 이동.
        if len(score_list) == 0 :
            move = [np.random.randint(move_cnt),0]
        if random.random() < explore_change:
            move = move_list[random.randint(0, len(move_list) - 1)]
        else:
            move = best_move
            
        self.update_explore_val()    
        
        return move
    
    def update_explore_val(self):
        #랜덤 이동 확률 점점 감소한다.
        if self.explore_change > 0.001:
            self.explore_change *= 0.99
        else : 
            self.explore_change = 0
            
    #학습 정책
    def train_policy(self,weights):
        """_학습 알고리즘
        1. 현재 상태에서 best action (놓을 열, 회전 횟수) 탐색.
        2. 

        Args:
            weights (_type_): _description_

        Returns:
            _type_: _description_
        """

        #action 탐색
        move = self.find_best_move(self.field, self.mino, weights, self.__get_explore_change__())
        current_params = self.get_state(self.field)
        test_field = copy.deepcopy(self.field)
        test_mino = copy.deepcopy(self.mino)
        test_field = self.simulate_map(test_field, test_mino, move)#(after_simul_map, reward)

        if test_field is not None:
            new_params = self.get_state(test_field[0])
            self.one_step_reward = test_field[1]        
        
        regularization_term = abs(sum(weights)) / len(weights)
        #regularization_term = abs(sum(weights))
                                                      
        
        #마코프 프로세스.
        for i in range(len(weights)):
            weights[i] = weights[i] + ALPHA *  (
                self.one_step_reward - current_params[i] + GAMMA * new_params[i])

        
        #정규화 함수들
        def standardscale(w):
            mean = abs(np.mean(w))
            std = np.std(w)
            if std != 0:
                w = [(x - mean) / std for x in w]
            return w
        
        def mynorm(w):
            for i in range(len(w)):
                w[i] = 100 * w[i] / regularization_term
                w[i] = math.floor(1e4 * w[i]) / 1e4  # Rounds the weights
            return w
        
        def l2_norm(w):
            norm_factor = np.sqrt(np.sum(np.square(w)))
            normalized_weights = w / norm_factor
            return normalized_weights
        #오버피팅을 피하기 위해 가중치를 정규화
        
        #weights = standardscale(weights)
        weights = mynorm(weights)
        
        return move, weights
    
    
    def choose_action(self):
        #print('현재필드')
        #self.show_field()
        
        move, weights = self.train_policy(self.weights)
        return move, weights
    
    def __get_explore_change__(self):
        return self.explore_change
    
    def __get_weights__(self):
        return self.weights
    
    def __set_weights__(self,weights):
        self.weights = weights