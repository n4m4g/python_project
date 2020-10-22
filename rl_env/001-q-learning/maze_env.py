#!/usr/bin/python3

import time
import numpy as np
import tkinter as tk

UNIT = 80
H, W = 5, 5

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['E', 'W', 'S', 'N']
        self.n_action = len(self.action_space)
        self.title('maze')
        self.geometry("{}x{}".format(UNIT*H, UNIT*W))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self,
                        bg='white',
                        height=UNIT*H,
                        width=UNIT*W)

        for h in range(1, H):
            x0, y0, x1, y1 = UNIT*h, 0, UNIT*h, UNIT*H
            self.canvas.create_line(x0, y0, x1, y1)

        for w in range(1, W):
            x0, y0, x1, y1 = 0, UNIT*w, UNIT*W, UNIT*w
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([UNIT//2, UNIT//2])

        h1_c = origin + np.array([UNIT*2, UNIT*1])
        self.h1 = self.canvas.create_rectangle(
                h1_c[0] - (UNIT*3//8), h1_c[1] - (UNIT*3//8),
                h1_c[0] + (UNIT*3//8), h1_c[1] + (UNIT*3//8),
                fill='black')

        h2_c = origin + np.array([UNIT*1, UNIT*2])
        self.h2 = self.canvas.create_rectangle(
                h2_c[0] - (UNIT*3//8), h2_c[1] - (UNIT*3//8),
                h2_c[0] + (UNIT*3//8), h2_c[1] + (UNIT*3//8),
                fill='black')

        oval_c = origin + np.array([UNIT*2, UNIT*2])
        self.oval = self.canvas.create_oval(
                oval_c[0] - (UNIT*3//8), oval_c[1] - (UNIT*3//8),
                oval_c[0] + (UNIT*3//8), oval_c[1] + (UNIT*3//8),
                fill='yellow')

        rect_c = origin
        self.rect = self.canvas.create_rectangle(
                rect_c[0] - (UNIT*3//8), rect_c[1] - (UNIT*3//8),
                rect_c[0] + (UNIT*3//8), rect_c[1] + (UNIT*3//8),
                fill='red')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([UNIT//2, UNIT//2])
        rect_c = origin
        self.rect = self.canvas.create_rectangle(
                rect_c[0] - (UNIT*3//8), rect_c[1] - (UNIT*3//8),
                rect_c[0] + (UNIT*3//8), rect_c[1] + (UNIT*3//8),
                fill='red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0,0])

        if action == 0: # east
            if s[0] < (W-1)*UNIT:
                base_action[0] += UNIT
        elif action == 1: # west
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 2: # south
            if s[1] < (H-1)*UNIT:
                base_action[1] += UNIT
        elif action == 3: # north
            if s[1] > UNIT:
                base_action[1] -= UNIT
        else:
            assert False, 'invalid action'

        self.canvas.move(self.rect, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.rect)

        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.h1), self.canvas.coords(self.h2)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.05)
        self.update()

def main():
    env = Maze()
    env.mainloop()

if __name__ == "__main__":
    main()

