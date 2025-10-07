# ML-Solve-Sudoku
ML + Backtracking For Solve Sudoku Problem Each Difficult [Easy, Medium, Hard]

--How to Use---
1. run python file test => python test_model_on_gui.py
2. select difficult
3. create new board (สร้างบอร์ดใหม่)
4. solve with ml (เริ่มแก้)
5. wait for success
6. check accurancy of solve with ml

---How to Train---
1. change amount your board will train (n_samples)
2. change difficulty ratio for board (p) ex. p = [0.1,0.2,0.7] mean => easy 10%, medium 20%, hard 70%
3. change epochs in model.fit
4. another setting as model.Sequential(Layer), model.compile(learning_rate), model.fit(batch_size, validation_split, reduce_lr)
5. run python file train => python train_v2.py