#train_v2.py
#credit by TM-view
import numpy as np
import os
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================
# ฟังก์ชันสร้างบอร์ดแบบ vectorized
# ======================
np.random.seed()
base = 3
side = base * base

def pattern(r, c): 
    return (base*(r % base) + r//base + c) % side

def generate_full_board():
    r_base = np.arange(base)
    rows  = np.r_[base*np.random.permutation(r_base)[:, None] + np.random.permutation(r_base)].flatten()
    cols  = np.r_[base*np.random.permutation(r_base)[:, None] + np.random.permutation(r_base)].flatten()
    nums  = np.random.permutation(np.arange(1, side+1))
    board = nums[pattern(rows[:, None], cols)]
    return board.astype(np.uint8)

def remove_cells(board, difficulty="medium"):
    show = board.copy()
    if difficulty=="easy":
        n_remove = np.random.randint(10,20)
    elif difficulty=="hard":
        n_remove = np.random.randint(55,65)
    else:  # medium
        n_remove = np.random.randint(35,45)
    idx = np.random.choice(81, n_remove, replace=False)
    show.flat[idx] = 0
    return show

# ======================
# CNN Model
# ======================
def build_model():
    model = models.Sequential([
        layers.Input((9,9,1)),
        
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(81*9, activation='softmax'),
        layers.Reshape((9,9,9))
    ])
    model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
    return model

# ======================
# Sudoku-aware Augmentation
# ======================
def augment_sudoku(board):
    """ทำ data augmentation ที่คงความถูกต้องของ Sudoku"""
    b = board.copy()

    # 1️⃣ Random permutation ของตัวเลข 1-9
    digits = np.arange(1, 10)
    perm = np.random.permutation(digits)
    for i, d in enumerate(digits, start=1):
        b[board == d] = perm[i - 1]

    # 2️⃣ สลับแถวภายใน band (3 แถวในแต่ละบล็อกแนวนอน)
    for band in range(3):
        rows = np.arange(band * 3, band * 3 + 3)
        np.random.shuffle(rows)
        b[band * 3:band * 3 + 3] = b[rows]

    # 3️⃣ สลับคอลัมน์ภายใน stack (3 คอลัมน์ในแต่ละบล็อกแนวตั้ง)
    for stack in range(3):
        cols = np.arange(stack * 3, stack * 3 + 3)
        np.random.shuffle(cols)
        b[:, stack * 3:stack * 3 + 3] = b[:, cols]

    # 4️⃣ สลับ band ทั้งชุด (แนวนอน)
    band_order = np.random.permutation(3)
    b = b[np.r_[band_order[0]*3:band_order[0]*3+3,
                band_order[1]*3:band_order[1]*3+3,
                band_order[2]*3:band_order[2]*3+3], :]

    # 5️⃣ สลับ stack ทั้งชุด (แนวตั้ง)
    stack_order = np.random.permutation(3)
    b = b[:, np.r_[stack_order[0]*3:stack_order[0]*3+3,
                   stack_order[1]*3:stack_order[1]*3+3,
                   stack_order[2]*3:stack_order[2]*3+3]]

    # 6️⃣ Random transpose (บางครั้ง)
    if np.random.rand() < 0.5:
        b = b.T

    return b

# ======================
# Backtracking + ML Solver
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
    if not empty: return True
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

def compute_accuracy(solved_board, solution, puzzle):
    correct, total = 0,0
    for i in range(9):
        for j in range(9):
            if puzzle[i,j]==0:
                total+=1
                if solved_board[i,j]==solution[i,j]:
                    correct+=1
    if total==0: return 1.0
    return correct/total

# ======================
# Callback เซฟโมเดลทุก epoch
# ======================
class SaveModelCallback(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath)
        print(f"\n💾 Model saved after epoch {epoch+1} -> {self.filepath}")

# ======================
# Main
# ======================
if __name__=="__main__":
    n_samples = 1000000
    X_train, Y_train = [], []

    try:
        print("🔹 Generating dataset...")
        for i in range(n_samples):
            solution = generate_full_board()
            difficulty = np.random.choice(["easy","medium","hard"], p=[0.01,0.19,0.8])
            puzzle = remove_cells(solution, difficulty=difficulty)
            
            solution_aug = augment_sudoku(solution)
            puzzle_aug = augment_sudoku(puzzle)
            
            X_train.append(puzzle)
            Y_train.append(solution)
            if (i+1) % 10000 == 0:
                print(f"✅ Generated {i+1}/{n_samples} boards")
    except KeyboardInterrupt:
        print("\n⚠️ KeyboardInterrupt: Saving dataset before exit...")

    # แปลงเป็น array + normalize
    X_train = np.array(X_train).reshape(-1,9,9,1)/9.0
    Y_train = to_categorical(np.array(Y_train)-1,num_classes=9)

    # Save dataset
    np.save("X_train.npy", X_train)
    np.save("Y_train.npy", Y_train)
    print("💾 Dataset saved: X_train.npy, Y_train.npy")

    # ======================
    # โหลดโมเดลถ้ามี, ถ้าไม่มีสร้างใหม่ (.keras format)
    # ======================
    model_file = "sudoku_model.keras"
    if os.path.exists(model_file):
        print(f"🔹 Loading existing model from {model_file}")
        model = models.load_model(model_file)
    else:
        print("🔹 Building new model")
        model = build_model()

    checkpoint = SaveModelCallback(model_file)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    # ======================
    # Train model
    # ======================
    try:
        print("🔹 Training model...")
        model.fit(X_train, Y_train, epochs=20, batch_size=16, validation_split=0.05, callbacks=[checkpoint, early_stop, reduce_lr], verbose=1)
    except KeyboardInterrupt:
        print("\n⚠️ KeyboardInterrupt: Saving model before exit...")
        model.save(model_file)
        print(f"💾 Model saved: {model_file}")
    finally:
        model.save(model_file)
        print(f"💾 Model saved: {model_file}")