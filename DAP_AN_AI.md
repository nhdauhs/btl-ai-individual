# BẢN ĐÁP ÁN BÀI TẬP LỚN AI

## Đề tài

Xây dựng chương trình chơi Tic-Tac-Toe/Caro trong đó người chơi hoặc máy tính
có thể sử dụng Minimax kết hợp cắt tỉa Alpha-Beta và MCTS kết hợp UCT.
Mục tiêu là tạo ra chương trình chơi đúng luật, có thể giải thích được quyết
định và kiểm chứng thực nghiệm hiệu quả của từng thuật toán.

> **Lưu ý:** Các số liệu trong bảng kết quả phải được điền sau khi chạy chương
> trình. Không nên tự tạo số liệu khi chưa thực nghiệm.

## Tóm tắt

Bài làm xây dựng một tác tử chơi trò chơi 5 quân liên tiếp. Hai phương pháp
được cài đặt là Minimax với cắt tỉa Alpha-Beta và Monte Carlo Tree Search
(MCTS) sử dụng công thức Upper Confidence Bound applied to Trees (UCT).
Minimax đánh giá các trạng thái bằng hàm lượng giá, còn MCTS lựa chọn nước đi
dựa trên nhiều lần mô phỏng. Thực nghiệm cần so sánh thời gian, số nút đã duyệt,
tỷ lệ thắng, hòa, thua và ảnh hưởng của hằng số khám phá `c`. Ngoài rollout
ngẫu nhiên, bài làm có thể dùng rollout heuristic để ưu tiên nước thắng ngay,
chặn đối thủ và tạo chuỗi quân liên tiếp. Kết quả kỳ vọng cho thấy Alpha-Beta
giảm đáng kể số nút phải xét, còn MCTS cải thiện dần khi tăng số vòng lặp.

## 1. Phát biểu bài toán

- Bàn cờ là ma trận kích thước `n × n`; có thể dùng bàn `3 × 3` để kiểm thử
  nhanh hoặc bàn lớn hơn cho luật **5 quân liên tiếp**.
- Hai người chơi lần lượt đặt quân `X` và `O` vào ô trống.
- Người tạo được một dãy ít nhất 5 quân liên tiếp theo hàng ngang, dọc hoặc
  đường chéo là người thắng.
- Nếu bàn cờ đầy mà không có người thắng, kết quả là hòa.
- Trạng thái cần lưu: bàn cờ, người đang đi, danh sách nước đi hợp lệ và kết quả
  hiện tại.

Hàm kiểm tra thắng phải xét bốn hướng:

```text
(0, 1)   ngang
(1, 0)   dọc
(1, 1)   chéo chính
(1, -1)  chéo phụ
```

Từ mỗi quân vừa đặt, đếm liên tiếp theo cả hai chiều. Nếu tổng số quân liên
tiếp đạt từ 5 trở lên thì kết thúc ván.

## 2. Minimax và cắt tỉa Alpha-Beta

### 2.1. Ý tưởng

Minimax giả sử cả hai bên đều chơi tối ưu:

- Ở lượt máy tính, chọn trạng thái có điểm lớn nhất (`MAX`).
- Ở lượt đối thủ, chọn trạng thái có điểm nhỏ nhất (`MIN`).
- Khi gặp trạng thái kết thúc, trả về điểm thắng, thua hoặc hòa.
- Khi không thể duyệt tới trạng thái kết thúc, dùng hàm lượng giá.

Quy ước điểm:

```text
máy thắng       +INF hoặc +1 000 000
đối thủ thắng   -INF hoặc -1 000 000
hòa              0
trạng thái chưa kết thúc: dùng heuristic
```

### 2.2. Hàm lượng giá đề xuất

Hàm lượng giá tính điểm cho từng đoạn có độ dài 5:

```text
Nếu đoạn chứa cả X và O:             0
Nếu có k quân máy, không có đối thủ:  score[k]
Nếu có k quân đối thủ, không có máy: -score[k]
```

Một bộ điểm đơn giản:

```text
score[1] = 1
score[2] = 10
score[3] = 100
score[4] = 10 000
score[5] = 1 000 000
```

Có thể cộng thêm điểm cho nước tạo hai hướng tấn công đồng thời và trừ điểm
cho nước để đối thủ có cơ hội thắng ngay. Hàm lượng giá chỉ là xấp xỉ nên cần
ghi rõ trong báo cáo rằng nó không thay thế việc kiểm tra trạng thái kết thúc.

### 2.3. Cắt tỉa Alpha-Beta

`alpha` là điểm tốt nhất mà MAX đã chắc chắn đạt được; `beta` là điểm tốt
nhất mà MIN đã chắc chắn đạt được. Khi `alpha >= beta`, các nhánh còn lại
không thể làm thay đổi quyết định nên được bỏ qua.

```text
Minimax(state, depth, alpha, beta, maximizing):
    nếu state kết thúc hoặc depth = 0:
        trả về evaluate(state)

    nếu maximizing:
        value = -INF
        cho mỗi nước đi hợp lệ:
            child = apply(state, nước đi)
            value = max(value, Minimax(child, depth-1,
                                       alpha, beta, false))
            alpha = max(alpha, value)
            nếu alpha >= beta: break
        trả về value

    ngược lại:
        value = +INF
        cho mỗi nước đi hợp lệ:
            child = apply(state, nước đi)
            value = min(value, Minimax(child, depth-1,
                                       alpha, beta, true))
            beta = min(beta, value)
            nếu alpha >= beta: break
        trả về value
```

Ở nút gốc, lưu nước đi có điểm Minimax tốt nhất. Độ sâu nên giới hạn để tránh
bùng nổ số trạng thái; có thể ưu tiên các nước gần quân đã có, nước thắng ngay
và nước phải chặn đối thủ.

## 3. MCTS và UCT

MCTS lặp bốn bước:

1. **Selection:** từ nút gốc đi xuống bằng UCT.
2. **Expansion:** mở rộng một nút con chưa được xét.
3. **Simulation/Rollout:** chơi tiếp đến khi kết thúc bằng ngẫu nhiên hoặc
   heuristic.
4. **Backpropagation:** truyền kết quả mô phỏng ngược về nút gốc.

### 3.1. Công thức UCT

Với nước đi `i`, điểm UCT là:

```text
UCT(i) = W_i / N_i + c * sqrt( ln(N_parent) / N_i )
```

Trong đó:

- `W_i`: tổng điểm thắng của các mô phỏng đi qua nhánh `i`.
- `N_i`: số lần nhánh `i` được thử.
- `N_parent`: số lần nút cha được thăm.
- `W_i / N_i`: thành phần **khai thác**, ưu tiên nhánh đang có kết quả tốt.
- `sqrt(ln(N_parent) / N_i)`: thành phần **khám phá**, ưu tiên nhánh ít thử.
- `c`: hằng số cân bằng khai thác và khám phá; giá trị lý thuyết thường gần
  `sqrt(2) ≈ 1.414`, nhưng cần kiểm tra bằng thực nghiệm.

Nếu `N_i = 0`, chọn nhánh đó ngay, không tính công thức để tránh chia cho 0.

### 3.2. Ý nghĩa căn bậc hai và logarit

UCT xuất phát từ bài toán Multi-Armed Bandit: mỗi nước đi được xem như một
"cánh tay", mỗi rollout trả về phần thưởng `R ∈ [0, 1]`, và mục tiêu là ước
lượng nước có giá trị kỳ vọng lớn nhất với số lần mô phỏng hữu hạn.

Gọi `μ_i` là giá trị thật của nước `i`, còn:

```text
X̄_i = W_i / N_i
```

là trung bình quan sát được sau `N_i` lần thử. Theo bất đẳng thức Hoeffding,
với mọi `ε > 0`:

```text
P(μ_i > X̄_i + ε) <= exp(-2 * N_i * ε^2)
```

Ta muốn sai số trên nhỏ khi nút cha đã được thăm nhiều lần. Chọn mức sai số
xác suất giảm theo số lần thăm, chẳng hạn `δ = N_parent^(-4)`, rồi giải:

```text
exp(-2 * N_i * ε^2) = δ

=> -2 * N_i * ε^2 = ln(δ)
=> ε^2 = ln(1 / δ) / (2 * N_i)
=> ε = sqrt(ln(1 / δ) / (2 * N_i))
```

Vì `δ = N_parent^(-4)` nên `ln(1 / δ) = 4 ln(N_parent)`. Do đó:

```text
ε = sqrt(2 * ln(N_parent) / N_i)
```

Ta nhận được một cận trên tin cậy cho giá trị thật:

```text
UCB_i = X̄_i + sqrt(2 * ln(N_parent) / N_i)
```

Đây là cơ sở của công thức UCB1. Trong cây trò chơi, mỗi nút con lại là một
bài toán Bandit tại nút cha, nên UCB1 được áp dụng ở mọi nút và gọi là UCT.
Thay hệ số lý thuyết `sqrt(2)` bằng tham số điều chỉnh `c` cho ta công thức
thực nghiệm:

```text
UCT(i) = W_i / N_i + c * sqrt(ln(N_parent) / N_i)
```

Như vậy:

- `ln(N_parent)`: khi nút cha được thăm thêm, các nhánh bị bỏ quên vẫn được
  khám phá, nhưng mức ưu tiên tăng chậm.
- `1 / N_i`: nhánh càng ít được thử thì độ không chắc chắn càng lớn và phần
  khám phá càng cao.
- `sqrt(...)`: chuyển cận xác suất thành độ lệch chuẩn/cỡ sai số; nó làm mức
  phạt tăng vừa phải thay vì tăng tuyến tính.
- `c`: hệ số điều chỉnh giữa khai thác và khám phá. Với cách suy ra Hoeffding
  ở trên, `c = sqrt(2) ≈ 1.414`; trong thực tế nên thử các giá trị `0.5,
  1.0, 1.4, 2.0`.

Đây là một cận xác suất chứ không phải công thức bảo đảm luôn chọn nước thắng
trong mọi số vòng lặp hữu hạn. Khi số lần mô phỏng tăng, trung bình `W_i/N_i`
ổn định hơn và các nhánh có giá trị thấp dần bị loại khỏi lựa chọn thực tế.

### 3.3. Pseudocode MCTS

```text
MCTS(root, iterations):
    lặp iterations lần:
        node = root

        trong khi node đã mở rộng hết và chưa kết thúc:
            node = child có UCT lớn nhất

        nếu node chưa kết thúc và còn nước chưa mở rộng:
            node = mở rộng một nước chưa thử

        result = rollout(node.state)

        trong khi node khác rỗng:
            node.visits += 1
            node.wins += result quy về góc nhìn của node
            node = node.parent

    trả về con của root có visits lớn nhất
```

Nên chọn con có `N_i` lớn nhất khi kết thúc tìm kiếm. Cách này ổn định hơn
chọn `W_i/N_i` lớn nhất, vì một nhánh chỉ được thử một lần có thể có tỷ lệ
thắng 100% do may mắn, trong khi nhánh được thử nhiều lần cho ước lượng đáng
tin cậy hơn.

## 3.4. Mã nguồn chạy được

File `tic_tac_toe_ai.py` cài đặt đầy đủ các thành phần chính:

- Bàn cờ kích thước tùy chọn và điều kiện 5 quân liên tiếp.
- Sinh nước đi quanh các quân đã có để giảm branching factor.
- Minimax giới hạn độ sâu và cắt tỉa Alpha-Beta.
- MCTS gồm Selection, Expansion, Simulation và Backpropagation.
- UCT với tham số `--exploration` tương ứng với `c`.
- Rollout ngẫu nhiên hoặc heuristic (`--heuristic`).
- Đếm số nút, số lần cắt tỉa và số lượt thăm gốc.

Chạy MCTS với rollout ngẫu nhiên:

```bash
python tic_tac_toe_ai.py --mode mcts --size 9 --iterations 1000
```

Chạy MCTS với rollout heuristic và `c = 1.4`:

```bash
python tic_tac_toe_ai.py --mode mcts --size 9 --iterations 2000 \
  --exploration 1.4 --heuristic
```

Chạy Minimax + Alpha-Beta ở độ sâu 2:

```bash
python tic_tac_toe_ai.py --mode minimax --size 5 --depth 2
```

Trong chương trình, trường `wins` của một nút được tính theo góc nhìn của
người vừa đi đến nút đó. Vì thế khi một nút cha chọn con bằng UCT, tỷ lệ
`wins/visits` luôn là tỷ lệ có lợi cho người đang được quyền chọn tại nút cha.
Kết quả hòa được tính `0.5`, thắng tính `1` và thua tính `0`.

## 4. Rollout ngẫu nhiên và heuristic

### Rollout ngẫu nhiên

Từ trạng thái được chọn, lần lượt chọn ngẫu nhiên một nước hợp lệ cho đến khi
ván đấu kết thúc. Cách này dễ cài đặt và không thiên vị, nhưng cần nhiều vòng
lặp vì thông tin mỗi mô phỏng khá nhiễu.

### Rollout heuristic

Trước khi chọn ngẫu nhiên, áp dụng thứ tự ưu tiên:

1. Nếu có nước giúp máy thắng ngay, chọn nước đó.
2. Nếu đối thủ có nước thắng ngay, chọn nước chặn.
3. Ưu tiên nước tạo chuỗi 2, 3 hoặc 4 quân.
4. Ưu tiên ô gần quân đã có hoặc gần trung tâm.
5. Nếu không có ưu tiên nào, chọn ngẫu nhiên.

Rollout heuristic thường cho tín hiệu tốt hơn rollout thuần ngẫu nhiên, nhưng
có thể làm tăng thời gian mỗi mô phỏng và tạo thiên lệch. Đây là ý tưởng gần
với một value network: thay vì chơi ngẫu nhiên hoàn toàn, hệ thống dùng một
hàm đánh giá để ước lượng chất lượng trạng thái.

## 5. Thiết kế thực nghiệm

Mỗi cấu hình nên chạy ít nhất 50 ván, đổi bên đi trước giữa các ván và dùng
seed cố định khi cần tái lập. So sánh các cấu hình sau:

- `c ∈ {0.5, 1.0, 1.4, 2.0}`.
- `iterations ∈ {100, 500, 1000, 2000, 5000}`.
- Rollout ngẫu nhiên và rollout heuristic.
- Minimax và Minimax + Alpha-Beta ở cùng độ sâu.
- Đối thủ ngẫu nhiên, Minimax hoặc một tác tử khác.

Các độ đo cần ghi:

- Tỷ lệ thắng, hòa, thua.
- Thời gian suy nghĩ trung bình mỗi nước.
- Số nút đã sinh/đã duyệt.
- Số vòng mô phỏng MCTS.
- Ở nút gốc: `W`, `N`, `W/N` của các nước con.

### Bảng kết quả cần điền

| Thuật toán | Độ sâu/iterations | Thắng | Hòa | Thua | Thời gian TB | Số nút TB |
|---|---:|---:|---:|---:|---:|---:|
| Minimax | ... | ... | ... | ... | ... | ... |
| Alpha-Beta | ... | ... | ... | ... | ... | ... |
| MCTS-UCT | ... | ... | ... | ... | ... | ... |

| `c` | Iterations | Tỷ lệ thắng | Tỷ lệ hòa | Tỷ lệ thua |
|---:|---:|---:|---:|---:|
| 0.5 | ... | ... | ... | ... |
| 1.0 | ... | ... | ... | ... |
| 1.4 | ... | ... | ... | ... |
| 2.0 | ... | ... | ... | ... |

Vẽ đồ thị đường với trục `x` là số iterations và trục `y` là tỷ lệ thắng;
mỗi giá trị `c` là một đường. Vẽ thêm biểu đồ so sánh rollout ngẫu nhiên và
rollout heuristic.

## 6. Phân tích kết quả mẫu

Sau khi điền số liệu, phần thảo luận có thể viết theo các ý sau:

1. Alpha-Beta cho kết quả giống Minimax ở cùng bài toán và cùng độ sâu, nhưng
   duyệt ít nút hơn nhờ loại bỏ các nhánh không ảnh hưởng đến quyết định.
2. Khi tăng iterations, MCTS có thêm dữ liệu nên tỷ lệ chọn nước tốt thường
   tăng, nhưng thời gian tính toán cũng tăng.
3. `c` nhỏ thiên về khai thác; `c` lớn thiên về khám phá. Giá trị khoảng `1.4`
   là điểm khởi đầu hợp lý nhưng không bảo đảm tốt nhất cho mọi bàn cờ.
4. Chọn con có `N` lớn nhất sau cùng giúp giảm ảnh hưởng của các kết quả thắng
   ngẫu nhiên từ những nhánh mới thử ít lần.
5. Rollout heuristic có thể tăng tỷ lệ thắng với cùng số iterations, nhưng chi
   phí mỗi rollout cao hơn và kết quả phụ thuộc chất lượng heuristic.
6. Không nên kết luận “MCTS không bao giờ thua” chỉ từ vài ván. Cần kiểm tra
   với đối thủ tối ưu, nhiều seed, đổi lượt đi và số ván đủ lớn. Với bàn 5
   quân lớn, tuyên bố tuyệt đối còn phụ thuộc kích thước bàn, luật biên và ngân
   sách tính toán.

## 7. Phần bonus: bài toán thực tế

Có thể mở rộng thành bài toán **lập lịch giao hàng**:

- Trạng thái: các đơn hàng đã giao, vị trí hiện tại và thời gian hiện tại.
- Nước đi: chọn đơn hàng tiếp theo.
- Hàm mục tiêu: giảm tổng quãng đường, thời gian giao trễ và số xe sử dụng.
- Minimax phù hợp khi có đối thủ hoặc môi trường cạnh tranh; MCTS phù hợp khi
  không gian kế hoạch lớn và cần tìm nghiệm tốt trong thời gian giới hạn.

Một hướng đơn giản hơn là xây dựng tác tử AI chọn tuyến đi trên lưới có vật
cản. MCTS mô phỏng các đường đi, rollout dùng khoảng cách Manhattan làm
heuristic, còn kết quả đo bằng độ dài đường đi và tỷ lệ đến đích.

## 8. Kết luận

Bài làm đã xác định mô hình trò chơi 5 quân liên tiếp và đề xuất hai nhóm
thuật toán. Minimax cho quyết định có cơ sở khi độ sâu được kiểm soát;
Alpha-Beta giúp giảm số trạng thái cần duyệt. MCTS-UCT linh hoạt hơn trên cây
lớn nhờ cân bằng khai thác và khám phá. Thực nghiệm cần được dùng để chọn `c`,
số iterations và kiểu rollout phù hợp thay vì chỉ dựa vào giá trị lý thuyết.
