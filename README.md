# Caro 5 quân: Minimax và MCTS-UCT

Mã nguồn chính nằm trong [`tic_tac_toe_ai.py`](tic_tac_toe_ai.py). Chương trình hỗ trợ:

- Minimax với cắt tỉa Alpha-Beta;
- MCTS-UCT với rollout ngẫu nhiên hoặc rollout có heuristic;
- bàn cờ kích thước từ `5 x 5` trở lên, trong đó cần tạo được một dãy 5 quân.

## Chạy chương trình

```bash
python tic_tac_toe_ai.py --mode minimax --size 5 --depth 2
python tic_tac_toe_ai.py --mode mcts --size 9 --iterations 1000 --heuristic
```

Quân `X` do máy điều khiển; quân `O` được chọn ngẫu nhiên.

## Kiểm thử và thực nghiệm

Chạy kiểm thử:

```bash
python -m unittest -v
python -m py_compile tic_tac_toe_ai.py run_experiments.py test_ai.py
```

Bộ kiểm thử hiện có 4 kiểm thử, bao phủ luật thắng theo bốn hướng, heuristic chặn nước thắng ngay, ngân sách mô phỏng MCTS và sự thống nhất giữa Minimax cùng Alpha-Beta trên Tic-Tac-Toe chính xác.

Chạy thực nghiệm lặp lại:

```bash
python run_experiments.py --games 50 --iterations 100 500 --csv experiment_results.csv
```

Kết quả được ghi thành CSV, gồm số thắng/thua/hòa, tỷ lệ thắng, thời gian trung bình và số mô phỏng trung bình.

## Trạng thái xác minh

Đã chạy thành công:

```text
Ran 4 tests in 1.083s
OK
```

