import tkinter as tk
from Tetris import Tetris
from Color import COLORS
from RLAgent_v1 import RLAgent_v1 as agent  # Make sure you import the TetrisGUI class
import threading
import time
import pickle
from datetime import datetime

class Application_train(tk.Frame):
    
    def __init__(self, iterations = 1000, master=None):
        self.weights = [-1,-1,-1,2,2]
        super().__init__(master)
        self.tetris = agent(Tetris())
        self.tetris.__set_weights__(self.weights)
        self.pack()
        self.create_widgets()
        self.update_clock()
        self.init_val()#가중치 초기화
        
        # Start the RL agent's moves in a separate thread
        #self.agent_thread = threading.Thread(target=self.agent_moves)
        #self.agent_thread = threading.Thread(target=lambda: self.agent_moves(self.tetris))
        #self.agent_thread.start()
        # Start the RL agent's moves
        self.play_with_rl_agent(self.tetris)
    
    def init_val(self):
        self.MAX_GAMES = 0
        self.games_completed = 0
        self.scoreArray = []
        self.game_index_array = []
        
        self.explore_change = 0.5

    def update_clock(self):# after 메소드로 테트리스 레벨 업데이트
        #self.tetris_agent.move(1, 0)
        self.update()
        self.master.after(int(1000*(0.66**self.tetris.level)), self.update_clock)
        
    
    def create_widgets(self):# 위젯 생성.
        CELL_SIZE = 30
        self.canvas = tk.Canvas(self, height=CELL_SIZE*self.tetris.FIELD_HEIGHT, 
                                      width = CELL_SIZE*self.tetris.FIELD_WIDTH, bg="black", bd=0)
        self.canvas.focus_set()
        self.rectangles = [
            self.canvas.create_rectangle(c*CELL_SIZE, r*CELL_SIZE, (c+1)*CELL_SIZE, (r+1)*CELL_SIZE)
                for r in range(self.tetris.FIELD_HEIGHT) for c in range(self.tetris.FIELD_WIDTH)
        ]
        #단순히 팩 지정.
        self.canvas.pack(side="left")
        self.status_msg = tk.Label(self, anchor='w', width=11, font=("Courier", 24))
        self.status_msg.pack(side="top")
        self.game_over_msg = tk.Label(self, anchor='w', width=11, font=("Courier", 24), fg='red')
        self.game_over_msg.pack(side="top")
    
    def update(self):
        for i, _id in enumerate(self.rectangles):
            color_num = self.tetris.get_color(i//self.tetris.FIELD_WIDTH, i % self.tetris.FIELD_WIDTH)
            self.canvas.itemconfig(_id, fill=COLORS[color_num])
    
        self.status_msg['text'] = "Score: {}\nLevel: {}".format(self.tetris.score, self.tetris.level)
        self.game_over_msg['text'] = "GAME OVER.\n" if self.tetris.game_over else ""

    def agent_moves(self,agent):
        while not agent.game_over:
            # Play one step of the game with the RL agent
            #self.tetris.play_one_step()
            self.play_with_rl_agent(agent)
            self.update()
            self.master.after(50)  # Adjust the delay between agent moves as needed

    def train(self, iterations):
        for i in range(iterations):
            print(f"Training iteration {i + 1}/{iterations}")
            self.tetris = agent(Tetris())  # Reset the game environment
            self.tetris.__set_weights__(self.weights)# set weights 가중치 주입
            self.play_with_rl_agent(self.tetris)  # Train the agent
            self.weights = self.tetris.__get_weights__()  # Update the weights

        filename = self.get_filename_with_timestamp()
        self.save_weights_to_file(self.weights, file_name=filename)
        print(f"Weights saved to {filename}")

    def play_with_rl_agent(self, agent, delay=50):
        if not agent.game_over:
            action = agent.choose_action()
            rot, sideways = action

            # Rotate
            for _ in range(rot):
                agent.rotate(1)

            # Move left or right
            agent.move(0, sideways)

            # Drop to the bottom
            drop_flag = True
            while drop_flag:
                drop_flag = agent.move(1, 0)

            self.update()
            self.master.after(delay, lambda: self.play_with_rl_agent(agent, delay))
        else:
            print("Game Over")
            print(f'가중치 {self.weights}')

    def get_filename_with_timestamp(self, base_name='model_ver_01'):
        current_time = datetime.now()
        timestamp = current_time.strftime('%Y-%m-%d_%H-%M-%S')
        return f"{base_name}_{timestamp}.pkl"
    
    def save_weights_to_file(self, weights, file_name='weights.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights_from_file(self, file_name='weights.pkl'):
        with open(file_name+'.pkl', 'rb') as f:
            self.weights = pickle.load(f)
        return self.weights

a = True
root = tk.Tk()
app = Application_train(master=root)
app.load_weights_from_file('./model_ver_01_2023-05-02_13-13-29')
print(app.tetris.__get_weights__())
#app.train(iterations=10000)

#app.play_with_rl_agent()  # Add this line to make the RL agent play the game
app.mainloop()