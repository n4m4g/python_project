#!/usr/bin/python3

import tkinter as tk

class Example(tk.Tk, object):
    def __init__(self):
        super(Example, self).__init__()
        self.title('window')
        self.geometry('200x100')
        self.var = tk.StringVar()
        self.ON_HIT = False

    def build(self):
        # label
        self.l = tk.Label(self,
                textvariable=self.var,
                bg='green',
                font=('Arial',12),
                width=15,
                height=2
                )
        self.l.pack()

        # button
        self.b = tk.Button(self,
                text='hit me',
                width=15,
                height=2,
                command=self.hit_me
                )
        self.b.pack()

    def hit_me(self):
        next_text = '' if self.ON_HIT else 'you hit me'
        self.var.set(next_text)
        self.ON_HIT = not self.ON_HIT

def main():
    w = Example()
    w.build()
    w.mainloop()

if __name__ == "__main__":
    main()
