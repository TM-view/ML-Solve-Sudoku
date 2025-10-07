# genboards.py
import numpy as np

np.random.seed()
# -----------------------------
# สร้าง Sudoku board สมบูรณ์ 100%
# -----------------------------
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

# -----------------------------
# ฟังก์ชันลบช่องตามความยาก (vectorized)
# -----------------------------
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

# -----------------------------
# main: สร้าง dataset และแบ่ง train/val
# -----------------------------
def main(num=10000000, val_ratio=0.05):
    n_val = int(num * val_ratio)
    n_train = num - n_val

    train_overall = np.zeros((n_train, 2, 9, 9), dtype=np.uint8)
    val_overall = np.zeros((n_val, 2, 9, 9), dtype=np.uint8)

    for i in range(num):
        board = generate_full_board()
        difficulty = np.random.choice(["easy", "medium", "hard"], p=[0.05,0.40,0.55])
        puzzle = remove_cells(board, difficulty=difficulty)

        if i < n_train:
            train_overall[i] = [puzzle, board]
        else:
            val_overall[i - n_train] = [puzzle, board]

        if (i + 1) % 10000 == 0:
            print(f"✅ สร้างไปแล้ว {i + 1}/{num} บอร์ด")

    np.save("sudoku_dataset.npy", train_overall)
    np.save("val_dataset.npy", val_overall)
    print(f"✅ บันทึกไฟล์ sudoku_dataset.npy และ val_dataset.npy เรียบร้อย! ({n_train}/{n_val} บอร์ด)")

if __name__ == "__main__":
    main(2000000)  