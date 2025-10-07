import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import models

def generate_full_board():
    base = 3
    side = base * base

    def pattern(r, c): return (base*(r % base) + r//base + c) % side

    r_base = np.arange(base)
    rows  = np.r_[base*np.random.permutation(r_base)[:, None] + np.random.permutation(r_base)].flatten()
    cols  = np.r_[base*np.random.permutation(r_base)[:, None] + np.random.permutation(r_base)].flatten()
    nums  = np.random.permutation(np.arange(1, side+1))

    board = nums[pattern(rows[:, None], cols)]
    return board.astype(np.uint8)

def remove_cells(board, difficulty="medium"):
    show = board.copy()
    if difficulty == "easy":
        n_remove = np.random.randint(10, 20)
    elif difficulty == "hard":
        n_remove = np.random.randint(50, 60)
    else:  # medium
        n_remove = np.random.randint(30, 40)
    idx = np.random.choice(81, n_remove, replace=False)
    show.flat[idx] = 0
    return show

# ======================
# ฟังก์ชัน Backtracking + ML Solver
# ======================
def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i,j]==0:
                return (i,j)
    return None

def is_valid(board,row,col,num):
    if num in board[row]: return False
    if num in board[:,col]: return False
    sr, sc = 3*(row//3),3*(col//3)
    if num in board[sr:sr+3, sc:sc+3]: return False
    return True

def solve_with_ml(board, probs):
    empty = find_empty(board)
    if not empty: 
        return True
    row,col = empty
    candidates = list(range(1,10))
    prob_values = probs[row,col]
    candidates.sort(key=lambda x: prob_values[x-1], reverse=True)
    for num in candidates:
        if is_valid(board,row,col,num):
            board[row,col]=num
            if solve_with_ml(board, probs):
                return True
            board[row,col]=0
    return False

def predict_probs(model, board):
    input_data = board.reshape(1,9,9,1)/9.0
    probs = model.predict(input_data, verbose=0)[0]
    return probs

# ======================
# ฟังก์ชัน compute_accuracy ใหม่
# ======================
def compute_accuracy_new(solved_board, solution, puzzle):
    """
    ความแม่นยำ = ช่องว่างทั้งหมดตอนแรก / (ช่องว่างทั้งหมด + จำนวนครั้งที่เติมผิด)
    """
    total_empty = np.sum(puzzle == 0)
    mistakes = 0
    
    for i in range(9):
        for j in range(9):
            if puzzle[i,j] == 0 and solved_board[i,j] != solution[i,j]:
                print(f"เติมเลข {solved_board[i,j]} ซึ่งผิดที่จุด {i},{j} มันควรเป็นเลข {solution[i,j]}")
                mistakes += 1

    if total_empty + mistakes == 0:
        return 1.0, mistakes
    
    accuracy = total_empty / (total_empty + mistakes)
    return 100 * accuracy, mistakes

# ======================
# Load trained model (.keras)
# ======================
model_file = "sudoku_model.keras"
try:
    print(f"\nLoading model from {model_file}")
    model = models.load_model(model_file)
    print(f"✅ Loaded model Success")
except:
    raise FileNotFoundError(f"❌ Model file {model_file} not found. Train model first.")

# ======================
# ฟังก์ชันทดสอบ
# ======================
def test_sudoku(puzzle, solution):
    """
    puzzle: 9x9 numpy array (ช่องว่าง = 0)
    solution: 9x9 numpy array
    """
    probs = predict_probs(model, puzzle)
    solved = puzzle.copy()
    solve_with_ml(solved, probs)
    acc, mistakes = compute_accuracy_new(solved, solution, puzzle)
    
    num_empty = np.sum(puzzle == 0)
    
    # print("🔹 Puzzle:")
    # print(puzzle)
    # print("\n🔹 Solved:")
    # print(solved)
    
    print(f"\n🔹 ความแม่นยำ = {acc:.2f}%")
    print(f"🔹 จำนวนครั้งที่เติมผิด = {mistakes}/{num_empty}\n")
    average_persent.append(acc)
    
    if len(average_persent) % 100 == 0:
        print("กำลังคำนวณโปรดรอสักครู่")
        
def find_average(exited_acc):
    return sum(exited_acc) / len(exited_acc)

# ======================
# ตัวอย่างการใช้
# ======================
if __name__=="__main__":
    num = 10
    ask = []
    ans = []
    average_persent = []
    
    puzzles = np.zeros((num, 9, 9), dtype=np.uint8)
    solutions = np.zeros((num, 9, 9), dtype=np.uint8)
    
    print(f"เริ่มสร้างกระดานทดสอบ {num} กระดาน")
    for i in range(num):
        board = generate_full_board()
        difficulty = "medium"
        puzzle = remove_cells(board, difficulty=difficulty)
        
        puzzles[i] = puzzle
        solutions[i] = board
        
        # if (i + 1) % 10000 == 0:
        #     print(f"✅ สร้างไปแล้ว {i + 1}/{num} บอร์ด")
    
    print(f"โมเดลเริ่มแก้กระดาน")
    for i in range(num):
        test_sudoku(puzzles[i], solutions[i])
    
    print(f"🔹 ความแม่นยำเฉลี่ยของกระดานระดับ {difficulty} จำนวน {num} กระดาน = {find_average(average_persent):.2f}%\n")