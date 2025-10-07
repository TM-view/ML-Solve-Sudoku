#test_model_on_gui.py
#credit by TM-view
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import time
import os
from tensorflow.keras import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# 🔹 Sudoku Utility
# ============================================================
def generate_full_board():
    base = 3
    side = base * base
    def pattern(r, c): return (base*(r % base) + r//base + c) % side
    r_base = np.arange(base)
    rows = np.r_[base*np.random.permutation(r_base)[:, None] + np.random.permutation(r_base)].flatten()
    cols = np.r_[base*np.random.permutation(r_base)[:, None] + np.random.permutation(r_base)].flatten()
    nums = np.random.permutation(np.arange(1, side+1))
    board = nums[pattern(rows[:, None], cols)]
    return board.astype(np.uint8)

def remove_cells(board, difficulty="medium"):
    show = board.copy()
    if difficulty == "easy":
        n_remove = np.random.randint(10, 20)
    elif difficulty == "hard":
        n_remove = np.random.randint(55, 65)
    else:
        n_remove = np.random.randint(35, 45)
    idx = np.random.choice(81, n_remove, replace=False)
    show.flat[idx] = 0
    return show

def is_valid(board, row, col, num):
    if num in board[row]: return False
    if num in board[:, col]: return False
    sr, sc = 3*(row//3), 3*(col//3)
    if num in board[sr:sr+3, sc:sc+3]: return False
    return True

# ============================================================
# 🔹 ML Solver
# ============================================================
def predict_probs(model, board):
    inp = board.reshape(1, 9, 9, 1) / 9.0
    return model.predict(inp, verbose=0)[0]

def find_most_confident_empty(board, probs):
    max_conf = -1
    pos = None
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                conf = np.max(probs[i, j])
                if conf > max_conf:
                    max_conf = conf
                    pos = (i, j)
    return pos

# ============================================================
# 🔹 GUI-Based Visual Solver
# ============================================================
class SudokuGUI:
    def __init__(self, root, model_path="sudoku_model.keras"):
        self.root = root
        self.root.title("🧩 Sudoku Visualizer")
        self.root.geometry("1200x850")

        # โหลดโมเดล
        try:
            self.model = models.load_model(model_path)
        except:
            messagebox.showerror("Error", f"❌ ไม่พบโมเดล {model_path}")
            self.root.destroy()
            return

        self.board = None
        self.solution = None
        self.puzzle_labels = [[None]*9 for _ in range(9)]
        self.solution_labels = [[None]*9 for _ in range(9)]

        self.create_widgets()

    # --------------------------------------------------------
    def create_widgets(self):
        ttk.Label(self.root, text="Sudoku Solver Visualizer", font=("Arial", 18, "bold")).pack(pady=10)
        
        # Control Frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)
        ttk.Label(control_frame, text="ระดับความยาก:").grid(row=0, column=0)
        self.diff_var = tk.StringVar(value="medium")
        ttk.Combobox(control_frame, textvariable=self.diff_var, values=["easy", "medium", "hard"], width=10).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="🎲 สร้างบอร์ดใหม่", command=self.new_board).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="🚀 เริ่มแก้", command=self.start_solving).grid(row=0, column=3, padx=5)

        # Grids Frame
        grids_frame = ttk.Frame(self.root)
        grids_frame.pack(pady=15)

        # Puzzle Grid
        puzzle_frame = ttk.Frame(grids_frame)
        puzzle_frame.grid(row=0, column=0, padx=30)
        ttk.Label(puzzle_frame, text="Puzzle Board", font=("Arial",12,"bold")).grid(row=0,column=0,columnspan=9,pady=5)
        self.create_grid(puzzle_frame, self.puzzle_labels, bg_color="white")

        # Solution Grid
        solution_frame = ttk.Frame(grids_frame)
        solution_frame.grid(row=0, column=1, padx=30)
        ttk.Label(solution_frame, text="Solution Board", font=("Arial",12,"bold")).grid(row=0,column=0,columnspan=9,pady=5)
        self.create_grid(solution_frame, self.solution_labels, bg_color="lightyellow")

        self.status_label = ttk.Label(self.root, text="พร้อมทำงาน", font=("Arial", 12))
        self.status_label.pack(pady=10)

    # --------------------------------------------------------
    def create_grid(self, parent_frame, label_grid, bg_color="white"):
        for block_r in range(3):
            for block_c in range(3):
                block_frame = tk.Frame(parent_frame, relief="solid", borderwidth=2)
                block_frame.grid(row=block_r, column=block_c, padx=1, pady=1)
                for i in range(3):
                    for j in range(3):
                        r, c = block_r*3+i, block_c*3+j
                        label = tk.Label(block_frame, text="", width=4, height=2,
                                         font=("Consolas", 18, "bold"), relief="ridge", borderwidth=1,
                                         bg=bg_color)
                        label.grid(row=i, column=j, padx=1, pady=1)
                        label_grid[r][c] = label

    # --------------------------------------------------------
    def new_board(self):
        self.solution = generate_full_board()
        self.board = remove_cells(self.solution, self.diff_var.get())
        self.update_grid()
        self.status_label.config(text=f"สร้างบอร์ดใหม่ ({self.diff_var.get()}) แล้ว")

    # --------------------------------------------------------
    def update_grid(self, highlight=None):
        for i in range(9):
            for j in range(9):
                num = self.board[i, j]
                color = "white"
                if highlight and (i,j)==highlight[0]:
                    color = highlight[1]
                self.puzzle_labels[i][j].config(text=str(num) if num !=0 else "", bg=color)
                self.solution_labels[i][j].config(text=str(self.solution[i,j]))

    # --------------------------------------------------------
    def start_solving(self):
        if self.board is None:
            messagebox.showwarning("⚠️", "กรุณาสร้างบอร์ดก่อน")
            return
        self.status_label.config(text="🧩 กำลังแก้บอร์ด...")
        threading.Thread(target=self.solve_step_by_step, daemon=True).start()

    # --------------------------------------------------------
    def solve_step_by_step(self):
        board = self.board.copy()
        step_delay = 0.05
        wrong_history = {}
        total_empty = np.sum(board==0)
        mistakes = 0

        while True:
            probs = predict_probs(self.model, board).reshape((9,9,9))
            for (i,j), wrong_set in wrong_history.items():
                for d in wrong_set:
                    probs[i,j,d-1] = 0

            empty = find_most_confident_empty(board, probs)
            if not empty:
                break
            r, c = empty
            best_digit = np.argmax(probs[r,c]) +1

            if best_digit == self.solution[r,c]:
                board[r,c] = best_digit
                self.board[r,c] = best_digit
                self.update_grid(highlight=((r,c), "#90EE90"))
            else:
                if (r,c) not in wrong_history:
                    wrong_history[(r,c)] = set()
                wrong_history[(r,c)].add(best_digit)
                mistakes += 1
                self.update_grid(highlight=((r,c), "#FF7F7F"))

            time.sleep(step_delay)
            self.root.update()

        accuracy = total_empty / (total_empty + mistakes) * 100 if (total_empty + mistakes)>0 else 0
        self.status_label.config(text=f"✅ เสร็จสิ้น | Accuracy: {accuracy:.2f}% | Mistakes: {mistakes}/{total_empty}")

# ============================================================
# 🔹 Run GUI
# ============================================================
if __name__=="__main__":
    root = tk.Tk()
    SudokuGUI(root)
    root.mainloop()
