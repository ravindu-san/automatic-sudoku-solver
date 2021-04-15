
import tkinter as tk
import tkinter.filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import cv2
from keras.models import load_model
import time as t

from project_results import project_to_original_img
from sudoku_grid import get_the_grid
from sudoku_solver import solve_2d


class SudokuApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, LiveFeedPage, ImageSelectPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#3d3d5c')
        self.controller = controller
        self.controller.title('Sudoku Solver')

        heading_label = tk.Label(self,
                                 text='Welcome To Sudoku Solver',
                                 font=('orbitron', 45, 'bold'),
                                 foreground='#ffffff',
                                 background='#3d3d5c')
        heading_label.pack(pady=25)

        space_label = tk.Label(self, height=4, bg='#3d3d5c')
        space_label.pack()

        select_method_label = tk.Label(self,
                                       text='Select a method',
                                       font=('orbitron', 13),
                                       bg='#3d3d5c',
                                       fg='white')
        select_method_label.pack(pady=10)

        def move_to_live_feed_solve():
            controller.show_frame('LiveFeedPage')

        def move_to_device_img_solve():
            controller.show_frame('ImageSelectPage')

        live_feed_button = tk.Button(self,
                                     text='Live Feed',
                                     command=move_to_live_feed_solve,
                                     relief='raised',
                                     borderwidth=3,
                                     width=40,
                                     height=3)
        live_feed_button.pack(pady=10)

        device_img_button = tk.Button(self,
                                      text='Device',
                                      command=move_to_device_img_solve,
                                      relief='raised',
                                      borderwidth=3,
                                      width=40,
                                      height=3)
        device_img_button.pack(pady=10)

        bottom_frame = tk.Frame(self, relief='raised', borderwidth=3)
        bottom_frame.pack(fill='x', side='bottom')

        def tick():
            current_time = t.strftime('%I:%M %p').lstrip('0').replace(' 0', ' ')
            time_label.config(text=current_time)
            time_label.after(200, tick)

        time_label = tk.Label(bottom_frame, font=('orbitron', 12))
        time_label.pack(side='right')

        tick()


class LiveFeedPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#3d3d5c')
        self.controller = controller
        self.cap = cv2.VideoCapture(0)
        # self.model = load_model('model/digit_classifier.h5')
        self.model = load_model('model/digit_model_ubuntu_fonts_imported.h5')

        heading_label = tk.Label(self,
                                 text='Live Feed',
                                 font=('orbitron', 45, 'bold'),
                                 foreground='#ffffff',
                                 background='#3d3d5c')
        heading_label.pack(pady=25)

        video_frame = tk.Frame(self, bg='#ffffff')
        video_frame.pack(fill='both', expand=True)

        button_frame = tk.Frame(self, bg='#33334d')
        button_frame.pack(fill='both', expand=True)

        vid1 = tk.Label(video_frame)
        vid1.pack()

        def start_vid():
            ret, frame = self.cap.read()

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            vid1.imgtk = imgtk
            vid1.configure(image=imgtk)

            grid_values = get_the_grid(frame, self.model)

            if grid_values is not None:
                warped_puzzle, puzzle_contour, crop_indices, board, print_list = grid_values
            else:
                vid1.after(1, start_vid)
                return

            if np.all(board == 0):
                vid1.after(1, start_vid)
                return

            solution = np.array(solve_2d(board)[1])

            if solution.size == 0:
                vid1.after(1, start_vid)
                return

            print_img = project_to_original_img(warped_puzzle, print_list, solution, frame, puzzle_contour, crop_indices)

            cv2image = cv2.cvtColor(print_img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            vid1.imgtk = imgtk
            vid1.configure(image=imgtk)
            vid1.after(1000, start_vid)

        start_button = tk.Button(button_frame,
                                 text='Start',
                                 command=start_vid,
                                 relief='raised',
                                 borderwidth=3,
                                 width=30,
                                 height=3)
        start_button.grid(row=3, column=2, pady=5)

        def back_to_start_page():
            controller.show_frame("StartPage")

        back_button = tk.Button(button_frame,
                                text='Back',
                                command=back_to_start_page,
                                relief='raised',
                                borderwidth=3,
                                width=30,
                                height=3)
        back_button.grid(row=3, column=1, pady=5)

        bottom_frame = tk.Frame(self, relief='raised', borderwidth=3)
        bottom_frame.pack(fill='x', side='bottom')

        time_label = tk.Label(bottom_frame, font=('orbitron', 12))
        time_label.pack(side='right')


class ImageSelectPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#3d3d5c')
        self.controller = controller
        self.original_img = None
        self.solved_img = None
        self.original_img_label = None
        self.solved_img_label = None
        # self.model = load_model('model/digit_classifier.h5')
        self.model = load_model('model/digit_model_ubuntu_fonts_imported.h5')

        heading_label = tk.Label(self,
                                 text='Device Image',
                                 font=('orbitron', 45, 'bold'),
                                 foreground='#ffffff',
                                 background='#3d3d5c')
        heading_label.pack(pady=25)

        images_label = tk.Label(self)
        images_label.pack(pady=25, expand=True)

        button_frame = tk.Frame(self, bg='#33334d')
        button_frame.pack(fill='both', expand=True)

        def back_to_start_page():
            set_default_input_img_label()
            set_default_output_img_label()
            result_time_label_2.config(text=0.000)
            controller.show_frame("StartPage")

        back_button = tk.Button(button_frame,
                                   text='Back',
                                   command=back_to_start_page,
                                   relief='raised',
                                   borderwidth=3,
                                   width=30,
                                   height=3)
        back_button.grid(row=3, column=1, pady=5)

        def set_default_input_img_label():
            self.original_img = ImageTk.PhotoImage(Image.open("1.jpg"))
            self.original_img_label = tk.Label(images_label, image=self.original_img, width=400, height=400,
                                               borderwidth=3)
            self.original_img_label.grid(row=0, column=0, columnspan=1, pady=5)

        def set_default_output_img_label():
            self.solved_img = ImageTk.PhotoImage(Image.open('1.jpg'))
            self.solved_img_label = tk.Label(images_label, image=self.solved_img, width=400, height=400, borderwidth=3)
            self.solved_img_label.grid(row=0, column=2, columnspan=1, pady=5)

        set_default_input_img_label()
        set_default_output_img_label()

        def solve_device_sudoku_board():

            set_default_input_img_label()
            set_default_output_img_label()
            result_time_label_2.config(text=0.000)

            file_name = tk.filedialog.askopenfilename(title='Select an Image', filetypes=(
            ("png files", "*.png"), ("jpg files", "*.jpg"), ("jpeg files", "*.jpeg")))
            original_img = Image.open(file_name)
            original_img = original_img.resize((400, 400), Image.ANTIALIAS)
            self.original_img = ImageTk.PhotoImage(original_img)

            self.original_img_label = tk.Label(images_label, image=self.original_img, width=400, height=400,
                                               borderwidth=3)
            self.original_img_label.grid(row=0, column=0, columnspan=1, pady=5)

            start_time = t.time()

            img = cv2.imread(file_name)

            grid_values = get_the_grid(img, self.model)

            if grid_values is not None:
                warped_puzzle, puzzle_contour, crop_indices, board, print_list = grid_values
            else:
                messagebox.showwarning(title=None, message="Cannot detect a grid!",)
                return

            if np.all(board == 0):
                messagebox.showwarning(title=None, message="Cannot recognize digits correctly!", )
                return

            solution = np.array(solve_2d(board)[1])

            if solution.size == 0:
                messagebox.showwarning(title=None, message="Extracted puzzle not satisfy the constraints!", )
                return

            print_img = project_to_original_img(warped_puzzle, print_list, solution, img, puzzle_contour, crop_indices)

            end_time = t.time()

            cv2image = cv2.cvtColor(print_img, cv2.COLOR_BGR2RGBA)
            solved_img = Image.fromarray(cv2image)
            solved_img = solved_img.resize((400, 400), Image.ANTIALIAS)
            self.solved_img = ImageTk.PhotoImage(solved_img)
            self.solved_img_label = tk.Label(images_label, image=self.solved_img, width=400, height=400, borderwidth=3)
            self.solved_img_label.grid(row=0, column=2, columnspan=1, pady=5)

            time_taken = end_time - start_time
            result_time_label_2.config(text=round(time_taken, 3))
            print("time :", time_taken)

        select_img_button = tk.Button(button_frame,
                                   text='Select Image',
                                   command=solve_device_sudoku_board,
                                   relief='raised',
                                   borderwidth=3,
                                   width=30,
                                   height=3)
        select_img_button.grid(row=3, column=2, pady=5)

        result_time_label_1 = tk.Label(button_frame, font=('orbitron', 12))
        result_time_label_1.grid(row=3, column=5, padx=100,pady=5)
        result_time_label_1.config(text="Time")

        result_time_label_2 = tk.Label(button_frame, font=('orbitron', 12))
        result_time_label_2.grid(row=3, column=6, pady=5)
        result_time_label_2.config(text=0.000)

        result_time_label_3 = tk.Label(button_frame, font=('orbitron', 12))
        result_time_label_3.grid(row=3, column=7, pady=5)
        result_time_label_3.config(text='s')

        bottom_frame = tk.Frame(self, relief='raised', borderwidth=3)
        bottom_frame.pack(fill='x', side='bottom')

        time_label = tk.Label(bottom_frame, font=('orbitron', 12))
        time_label.pack(side='right')


if __name__ == "__main__":
    app = SudokuApp()
    app.mainloop()
