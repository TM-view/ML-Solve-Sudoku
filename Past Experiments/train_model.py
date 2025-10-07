import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import signal
import sys
import os

# ========================
# Options
# ========================
train_from_scratch = False   # True = สร้างโมเดลใหม่, False = โหลดโมเดลเก่า
model_file = 'sudoku_cnn_model_v2.h5'
dataset_file = 'sudoku_dataset.npy'

# ========================
# 1. Data Generator (memory-efficient, no one-hot)
# ========================
class SudokuDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, npy_path, batch_size=256, indices=None, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.data = np.load(npy_path, mmap_mode='r')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        if indices is None:
            self.indices = np.arange(len(self.data))
        else:
            self.indices = indices
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = self.data[batch_idx]

        X = batch[:, 0].astype(np.float32) / 9.0
        X = X[..., np.newaxis]  # (batch, 9, 9, 1)
        y = batch[:, 1].astype(np.int32) - 1  # sparse labels 0-8
        
        if self.augment:
            X, y = self.apply_augmentation(X, y)
            
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def apply_augmentation(self, X, y):
        for i in range(len(X)):
            # -----------------------------
            # 1. Random rotation (0°, 90°, 180°, 270°)
            # -----------------------------
            k = np.random.randint(0, 4)  # 0,1,2,3
            if k > 0:
                X[i] = np.rot90(X[i], k=k, axes=(0,1))
                y[i] = np.rot90(y[i], k=k, axes=(0,1))

            # -----------------------------
            # 2. Random flip horizontal/vertical
            # -----------------------------
            if np.random.rand() < 0.5:
                X[i] = np.flip(X[i], axis=1)
                y[i] = np.flip(y[i], axis=1)
            if np.random.rand() < 0.5:
                X[i] = np.flip(X[i], axis=0)
                y[i] = np.flip(y[i], axis=0)

            # -----------------------------
            # 3. Swap row or column blocks (3x3 blocks)
            # -----------------------------
            # swap row blocks
            if np.random.rand() < 0.5:
                block1, block2 = np.random.choice(3, 2, replace=False)
                rows1 = slice(block1*3, block1*3+3)
                rows2 = slice(block2*3, block2*3+3)
                X[i][rows1], X[i][rows2] = X[i][rows2].copy(), X[i][rows1].copy()
                y[i][rows1], y[i][rows2] = y[i][rows2].copy(), y[i][rows1].copy()
            # swap column blocks
            if np.random.rand() < 0.5:
                block1, block2 = np.random.choice(3, 2, replace=False)
                cols1 = slice(block1*3, block1*3+3)
                cols2 = slice(block2*3, block2*3+3)
                X[i][:, cols1], X[i][:, cols2] = X[i][:, cols2].copy(), X[i][:, cols1].copy()
                y[i][:, cols1], y[i][:, cols2] = y[i][:, cols2].copy(), y[i][:, cols1].copy()
                
        return X, y
# ========================
# 2. Load dataset and split train/val
# ========================
dataset = np.load(dataset_file, mmap_mode='r')
num_boards = len(dataset)
indices = np.arange(num_boards)
np.random.shuffle(indices)

split = int(num_boards * 0.95)
train_idx = indices[:split]
val_idx = indices[split:]

train_gen = SudokuDataGenerator(dataset_file, batch_size=512, indices=train_idx, shuffle=True, augment=True)
val_gen   = SudokuDataGenerator(dataset_file, batch_size=512, indices=val_idx, shuffle=False, augment=False)

# ========================
# 3. Load or create model
# ========================
if train_from_scratch or not os.path.exists(model_file):
    print("🧠 Creating new model...")
    model = models.Sequential([
        layers.Conv2D(64, (3,3), padding='same', input_shape=(9,9,1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.15),

        layers.Conv2D(256, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(9, (1,1), activation='softmax', padding='same')
    ])
else:
    print(f"📂 Loading existing model from {model_file} ...")
    model = tf.keras.models.load_model(model_file)

try:
    import tensorflow_addons as tfa
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
except ImportError:
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ========================
# 4. Callbacks
# ========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Save model after every epoch
checkpoint = ModelCheckpoint(
    filepath=model_file,
    save_freq='epoch',
    save_weights_only=False,
    verbose=1
)

lr_schedule = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)
# ========================
# 5. Train
# ========================
def save_model_on_exit(sig, frame):
    model.save(model_file)
    print("✅ Cancel_Train -> Model saved")
    sys.exit(0)

# register signal handler
signal.signal(signal.SIGINT, save_model_on_exit)

try:
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=[early_stop, checkpoint, lr_schedule],
        shuffle = True,
    )
except Exception as e:
    print(f"\n❌ Exception occurred: {e}")
    print("Saving model before exit...")
    model.save(model_file)
    print("✅ Crash -> Model saved")
    raise
finally:
    # save final model at the end of normal training
    model.save(model_file)
    print(f"✅ Success -> Saved final model")
# ========================
# 6. Save final model
# ========================
model.save(model_file)