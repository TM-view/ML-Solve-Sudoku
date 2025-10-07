# test_model.py
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import sys
from tensorflow.keras.models import load_model

# ===== Load AI Model =====
model = load_model("sudoku_cnn_model_v2.h5")

# ===== Sudoku rules =====
def is_valid(board, row, col, num):
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    r0, c0 = (row // 3) * 3, (col // 3) * 3
    if num in board[r0:r0+3, c0:c0+3]:
        return False
    return True

# ===== GUI App =====
class SudokuGUI:
    def __init__(self, master):
        self.master = master
        master.title("Sudoku AI Solver")
        self.entries = []
        self.solution_entries = []
        self.board = np.zeros((9,9), dtype=int)
        self.solution = None
        self.use_solution = False

        self.create_grids()
        self.create_buttons()
        self.mode_selection()

        # Queue ของช่องที่ยังต้องเติม
        self.empty_cells = []       # ช่องว่างตอนเริ่ม
        self.retry_cells = []       # ช่องที่เติมผิดแล้วต้อง retry
        self.correct_count = 0
        self.step_count = 0
        self.wrong_c = 0
        self.wrong_counts = {}      # นับจำนวนครั้งที่ผิดต่อช่อง
        self.wrong_history = {}     # เก็บเลขที่ลองแล้วผิด ต่อช่อง

    # -------------------------
    # 1. สร้างกริด AI board และ Solution board
    # -------------------------
    def create_grids(self):
        self.canvas_size = 9*40 + 4*2
        self.left_offset = 0
        self.right_offset = self.canvas_size + 50

        # AI board
        self.ai_canvas = tk.Canvas(self.master, width=self.canvas_size, height=self.canvas_size)
        self.ai_canvas.place(x=self.right_offset, y=0)
        self.entries = []
        for i in range(9):
            row_entries = []
            for j in range(9):
                x = self.right_offset + j*40 + (j//3)*2
                y = i*40 + (i//3)*2
                e = tk.Entry(self.master, width=2, font=('Arial',18), justify='center')
                e.place(x=x, y=y, width=40, height=40)
                row_entries.append(e)
            self.entries.append(row_entries)
        for i in range(10):
            lw = 2 if i%3==0 else 1
            self.ai_canvas.create_line(0,i*40 + (i//3)*2,self.canvas_size,i*40 + (i//3)*2,width=lw)
            self.ai_canvas.create_line(i*40 + (i//3)*2,0,i*40 + (i//3)*2,self.canvas_size,width=lw)

        # Solution board
        self.sol_canvas = tk.Canvas(self.master, width=self.canvas_size, height=self.canvas_size)
        self.sol_canvas.place(x=self.left_offset, y=0)
        self.solution_entries = []
        for i in range(9):
            row_entries = []
            for j in range(9):
                x = j*40 + (j//3)*2
                y = i*40 + (i//3)*2
                e = tk.Entry(self.master, width=2, font=('Arial',18), justify='center', bg='lightyellow', state='readonly')
                e.place(x=x, y=y, width=40, height=40)
                row_entries.append(e)
            self.solution_entries.append(row_entries)
        for i in range(10):
            lw = 2 if i%3==0 else 1
            self.sol_canvas.create_line(0,i*40 + (i//3)*2,self.canvas_size,i*40 + (i//3)*2,width=lw)
            self.sol_canvas.create_line(i*40 + (i//3)*2,0,i*40 + (i//3)*2,self.canvas_size,width=lw)

    # -------------------------
    # 2. ปุ่ม
    # -------------------------
    def create_buttons(self):
        self.solve_btn = tk.Button(self.master, text="Solve Step-by-Step", command=self.solve_step_by_step)
        self.solve_btn.place(x=self.right_offset, y=self.canvas_size + 10)
        self.check_btn = tk.Button(self.master, text="Check Accuracy", command=self.accuracy)
        self.check_btn.place(x=self.right_offset + 150, y=self.canvas_size + 10)

    # -------------------------
    # 3. เลือกโหมดและ Solution
    # -------------------------
    def mode_selection(self):
        mode = messagebox.askyesno("Mode", "Do you want to fill the board manually? (Yes) Or input list? (No)")
        if not mode:
            user_input = simpledialog.askstring("Input Board", "Paste your 9x9 list (list of lists):")
            if user_input:
                self.board = np.array(eval(user_input))
                for i in range(9):
                    for j in range(9):
                        val = self.board[i,j]
                        if val != 0:
                            self.entries[i][j].insert(0,str(val))
                            self.entries[i][j].config(bg='lightgray')

        sol_mode = messagebox.askyesno("Solution", "Do you want to input solution for accuracy check?")
        if sol_mode:
            sol_input = simpledialog.askstring("Solution Board", "Paste your solution 9x9 list (list of lists):")
            if sol_input:
                self.solution = np.array(eval(sol_input))
                self.use_solution = True
                for i in range(9):
                    for j in range(9):
                        val = self.solution[i,j]
                        self.solution_entries[i][j].config(state='normal')
                        self.solution_entries[i][j].delete(0, tk.END)
                        self.solution_entries[i][j].insert(0,str(val))
                        self.solution_entries[i][j].config(state='readonly')

    # -------------------------
    # 4. อ่านค่าจาก GUI
    # -------------------------
    def read_board(self):
        for i in range(9):
            for j in range(9):
                val = self.entries[i][j].get()
                self.board[i,j] = int(val) if val.isdigit() else 0
        return self.board.copy()

    # -------------------------
    # 5. Solve Step-by-Step พร้อม retry
    # -------------------------
    def solve_step_by_step(self):
        self.board = self.read_board()
        self.pred = model.predict(self.board.reshape(1,9,9,1)/9.0, verbose=0)[0].reshape((9,9,9))

        # เตรียม queue ของช่องว่าง
        self.empty_cells = list(map(tuple,np.argwhere(self.board==0)))
        self.retry_cells = []
        
        self.step()
    
    def retry(self):
        self.solve_step_by_step()
        
    def accuracy(self):
        messagebox.askyesno("Accuracy", f"Wrong_Count = {self.wrong_c}\nAccuracy (correct / total guesses) = {self.correct_count}/{self.step_count} = {((self.correct_count / self.step_count) * 100):.2f}%")

    def step(self):
        candidates = [c for c in (self.empty_cells + self.retry_cells)]

        if not candidates:
            if self.use_solution:
                incorrect = [(i,j) for i in range(9) for j in range(9) if self.board[i,j] != self.solution[i,j]]
                if incorrect:
                    print(f"🔁 Retry round: {len(incorrect)} incorrect cells")
                    self.retry_cells = incorrect
                    self.pred = model.predict(self.board.reshape(1,9,9,1)/9.0, verbose=0)[0].reshape((9,9,9))
                    self.master.after(500, self.step)
                    return
                else:
                    print("✅ Sudoku solved correctly!")
                    self.accuracy()
                    sys.exit(0)
            return

        best_cell, best_digit, best_conf = None, None, -1
        for i,j in candidates:
            pred_copy = self.pred[i,j].copy()
            if (i,j) in self.wrong_history:
                for wrong_d in self.wrong_history[(i,j)]:
                    pred_copy[wrong_d - 1] = 0

            sorted_digits = np.argsort(pred_copy)[::-1]
            for d_idx in sorted_digits:
                d = d_idx + 1
                temp = self.board.copy()
                temp[i,j] = 0
                if is_valid(temp, i,j,d):
                    conf = pred_copy[d_idx]
                    if conf > best_conf:
                        best_conf = conf
                        best_cell, best_digit = (i,j), d
                    break
        
        self.step_count += 1

        i,j = best_cell
        self.board[i,j] = best_digit
        self.entries[i][j].delete(0, tk.END)
        self.entries[i][j].insert(0,str(best_digit))
                
        # self.read_board()
        # self.pred = model.predict(self.board.reshape(1,9,9,1)/9.0, verbose=0)[0].reshape((9,9,9))

        # ตรวจสอบเฉลย
        if self.solution is not None:
            if best_digit == self.solution[i,j]:
                self.entries[i][j].config(bg='lightgreen')
                if (i,j) in self.empty_cells: self.empty_cells.remove((i,j))
                if (i,j) in self.retry_cells: self.retry_cells.remove((i,j))
                print(f"✅ Correct ({i},{j}) = {best_digit}")
                self.correct_count += 1
            else:
                self.wrong_c += 1
                self.entries[i][j].config(bg='lightcoral')
                self.retry_cells.append((i,j)) if (i,j) not in self.retry_cells else None

                # บันทึกตัวเลขผิด
                if (i,j) not in self.wrong_history:
                    self.wrong_history[(i,j)] = set()
                self.wrong_history[(i,j)].add(best_digit)

                print(f"❌ Wrong ({i},{j}) = {best_digit} | Should be {self.solution[i,j]} (will avoid next time)")
                self.wrong_counts[(i,j)] = self.wrong_counts.get((i,j),0)+1
                if self.wrong_counts[(i,j)] >= 1:
                    for x in range(9):
                        for y in range(9):
                            if self.board[x,y] == 0 or (x,y) in self.retry_cells:
                                self.entries[x][y].delete(0, tk.END)
                    # self.retry()
                    # return   
                                
        else:
            self.entries[i][j].config(bg='lightblue')
            if (i,j) in self.empty_cells: self.empty_cells.remove((i,j))
            print(f"🧩 Filled ({i},{j}) = {best_digit}")

        self.read_board()
        self.pred = model.predict(self.board.reshape(1,9,9,1)/9.0, verbose=0)[0].reshape((9,9,9))
        self.master.after(50, self.step)

# ===== Run App =====
if __name__=="__main__":
    root = tk.Tk()
    root.geometry("900x450")
    app = SudokuGUI(root)
    root.mainloop()
