import tkinter as tk
from Tetris import Tetris
from Color import COLORS

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.tetris = Tetris()
        self.pack()
        self.create_widgets()
        self.update_clock()

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

root = tk.Tk()
app = Application(master=root)
app.mainloop()