# ML-Solve-Sudoku
ML + Backtracking For Solve Sudoku Problem Each Difficult [Easy, Medium, Hard]

--How to Use---
1. set your model_path for test as sudoku_model.keras
2. run python file test => python test_model_on_gui.py
3. select difficult
4. create new board (สร้างบอร์ดใหม่)
5. solve with ml (เริ่มแก้)
6. wait for success
7. check accurancy of solve with ml

---How to Train---
1. set your model_path to continues train as sudoku_model.keras (if unavailable system will create new model to you)
2. change amount your board will train (n_samples)
3. change difficulty ratio for board (p) ex. p = [0.1,0.2,0.7] mean => easy 10%, medium 20%, hard 70%
4. change epochs in model.fit
5. another setting as model.Sequential(Layer), model.compile(learning_rate), model.fit(batch_size, validation_split, reduce_lr)
6. run python file train => python train_v2.py

---How to Save Model---
1. autosave with SaveModelCallback will save your model every finish once epoch (exited in file)
2. Ctrl + C (Cancel Train) system will save your model before stop train
3. finish train as usual (finish every epochs)