import tkinter as tk
from Tetris import Tetris
from Color import COLORS
from RLAgent_v1 import RLAgent_v1 as agent  # Make sure you import the TetrisGUI class
import threading

"""

#class Application(tk.Frame):
class Application(agent):
    
    def __init__(self, master=None):
        super().__init__(master)
        self.tetris = Tetris()
        self.pack()
        self.create_widgets()
        self.update_clock()

        # Start the RL agent's moves in a separate thread
        self.agent_thread = threading.Thread(target=self.agent_moves)
        self.agent_thread.start()

    def update_clock(self):# after 메소드로 테트리스 레벨 업데이트
        self.tetris.move(1, 0)
        self.update()
        self.master.after(int(1000*(0.66**self.tetris.level)), self.update_clock)
    
    def create_widgets(self):# 위젯 생성.
        CELL_SIZE = 30
        self.canvas = tk.Canvas(self, height=CELL_SIZE*self.tetris.FIELD_HEIGHT, 
                                      width = CELL_SIZE*self.tetris.FIELD_WIDTH, bg="black", bd=0)
        self.canvas.bind('<Left>', lambda _: (self.tetris.move(0, -1), self.update()))
        self.canvas.bind('<Right>', lambda _: (self.tetris.move(0, 1), self.update()))
        self.canvas.bind('<Down>', lambda _: (self.tetris.move(1, 0), self.update()))
        self.canvas.bind('c', lambda _: (self.tetris.rotate(1), self.update()))
        self.canvas.bind('z', lambda _: (self.tetris.rotate(0), self.update()))
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

    def agent_moves(self):
        while not self.tetris.game_over:
            # Play one step of the game with the RL agent
            self.tetris.play_one_step()
            self.update()
            self.master.after(50)  # Adjust the delay between agent moves as needed

    def play_with_rl_agent(self):
        while not self.game_over:
            action = self.choose_action(self.get_state())
            if action == 0:  # Move left
                self.move(0, -1)
            elif action == 1:  # Move right
                self.move(0, 1)
            elif action == 2:  # Rotate
                self.rotate(1)

            self.move(1, 0)  # Move the mino down as part of the game loop
            self.master.after(50, self.draw)
            self.master.update()
        print("Game Over")
"""

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.user_tetris = Tetris()
        self.tetris_agent = agent(Tetris())
        self.pack()
        self.create_widgets()
        self.init_val()#변수 초기화
        self.update_clock()

        # Start the RL agent's moves in a separate thread
        #self.agent_thread = threading.Thread(target=self.agent_moves)
        #self.agent_thread.start()

    def init_val(self):
        self.MAX_GAMES = 0
        self.games_completed = 0
        self.scoreArray = []
        self.game_index_array = []
        self.weight0Array = []
        self.weight1Array = []
        self.weight2Array = []
        self.weight3Array = []
        self.weight4Array = []
        self.explore_change = 0.5

    def update_clock(self):
        self.user_tetris.move(1, 0)#테트리스를 움직입니다.
        if self.tetris_agent.game_over:#종료 기능
            return
        move, weight = self.tetris_agent.gradient_descent(self.tetris_agent.weights)
        #move = (rot, down)
        print('get moved')
        for i in range(move[0]):
            self.tetris_agent.rotate(1)
        for j in range(move[1]):
            self.tetris_agent.move(1,0)
                
        #newScore, weights, explore_change
        #self.tetris_agent.move(1,0) #agent 움직입니다. 
        #newScore, weights, explore_change = self.run_game(weights, explore_change)
        #print("Game Number ", self.games_completed, " achieved a score of: ", newScore)
        #self.scoreArray.append(newScore)
        #self.game_index_array.append(self.games_completed)
        #self.weight0Array.append(-weights[0])
        #self.weight1Array.append(-weights[1])
        #self.weight2Array.append(-weights[2])
        #self.weight3Array.append(-weights[3])
        #self.weight4Array.append(-weights[4])
        self.update()
        self.master.after(int(1000 * (0.66 ** self.user_tetris.level)), self.update_clock)

    def create_widgets(self):
        CELL_SIZE = 30

        # User Tetris canvas
        self.user_canvas = self.create_tetris_canvas(CELL_SIZE)
        self.user_canvas.bind('<Left>', lambda _: (self.user_tetris.move(0, -1), self.update()))
        self.user_canvas.bind('<Right>', lambda _: (self.user_tetris.move(0, 1), self.update()))
        self.user_canvas.bind('<Down>', lambda _: (self.user_tetris.move(1, 0), self.update()))
        self.user_canvas.bind('c', lambda _: (self.user_tetris.rotate(1), self.update()))
        self.user_canvas.bind('z', lambda _: (self.user_tetris.rotate(0), self.update()))
        self.user_canvas.focus_set()

        # RL Tetris canvas
        self.rl_canvas = self.create_tetris_canvas(CELL_SIZE)

        # Create labels and pack everything
        self.user_status_msg = tk.Label(self, anchor='w', width=11, font=("Courier", 24))
        self.user_status_msg.pack(side="top")
        self.rl_status_msg = tk.Label(self, anchor='w', width=11, font=("Courier", 24))
        self.rl_status_msg.pack(side="top")
        self.game_over_msg = tk.Label(self, anchor='w', width=11, font=("Courier", 24), fg='red')
        self.game_over_msg.pack(side="top")

    def create_tetris_canvas(self, cell_size):
        canvas = tk.Canvas(self, height=cell_size * Tetris.FIELD_HEIGHT,
                           width=cell_size * Tetris.FIELD_WIDTH, bg="black", bd=0)
        rectangles = [
            canvas.create_rectangle(c * cell_size, r * cell_size, (c + 1) * cell_size, (r + 1) * cell_size)
            for r in range(Tetris.FIELD_HEIGHT) for c in range(Tetris.FIELD_WIDTH)
        ]
        canvas.pack(side="left")
        canvas.rectangles = rectangles
        return canvas

    def agent_moves(self):
        while not self.tetris_agent.game_over:
            print('agent move..')
            self.tetris_agent.move(1,0)
            self.update()
            self.master.after(50)

    def update(self):
        self.update_canvas(self.user_canvas, self.user_tetris)
        self.update_canvas(self.rl_canvas, self.tetris_agent)

        self.user_status_msg['text'] = "User\nScore: {}\nLevel: {}".format(self.user_tetris.score, self.user_tetris.level)
        self.rl_status_msg['text'] = "Agent\nScore: {}\nLevel: {}".format(self.tetris_agent.score, self.tetris_agent.level)
        self.game_over_msg['text'] = "GAME OVER.\n" if self.user_tetris.game_over or self.tetris_agent.game_over else ""

    def update_canvas(self, canvas, tetris):
        for i, _id in enumerate(canvas.rectangles):
            color_num = tetris.get_color(i // tetris.FIELD_WIDTH, i % tetris.FIELD_WIDTH)
            canvas.itemconfig(_id, fill=COLORS[color_num])


root = tk.Tk()
app = Application(master=root)
#app.play_with_rl_agent()  # Add this line to make the RL agent play the game
app.mainloop()