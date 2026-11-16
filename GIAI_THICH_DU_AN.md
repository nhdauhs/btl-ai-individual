# Giải thích toàn bộ dự án Tic-Tac-Toe/Caro bằng AI

## 1. Mục đích của tài liệu

Thư mục này là một bài tập lớn về tìm kiếm đối kháng trong trí tuệ nhân tạo.
Mục tiêu là xây dựng chương trình chơi trò chơi hai người, rồi giải thích và
so sánh hai nhóm phương pháp:

1. **Minimax kết hợp cắt tỉa Alpha–Beta**: tìm kiếm có hệ thống trong cây trò
   chơi. Với Tic-Tac-Toe 3×3, cây đủ nhỏ để tìm đến trạng thái kết thúc và
   quyết định tối ưu tuyệt đối.
2. **Monte Carlo Tree Search (MCTS) kết hợp UCT**: xây dựng dần cây bằng các
   ván mô phỏng. Phương pháp này phù hợp hơn với Caro vì cây tìm kiếm lớn.

Tài liệu này giải thích mã nguồn, mô hình toán học, cách chạy, các file báo cáo,
độ phức tạp, quy trình thực nghiệm và các giới hạn hiện tại. Các con số hiệu
năng cụ thể phải được đo bằng chương trình; không nên suy ra số liệu thực nghiệm
từ công thức lý thuyết.

---

## 2. Sơ đồ nội dung của thư mục

| File | Vai trò |
|---|---|
| `tic_tac_toe_ai.py` | Chương trình Caro 5 quân với Minimax–Alpha-Beta hoặc MCTS–UCT. |
| `ttt_minimax.py` | Chương trình Tic-Tac-Toe 3×3, máy dùng Minimax hoặc Alpha-Beta, người chơi là O. |
| `DAP_AN_AI.md` | Bản đáp án và khung nội dung báo cáo bằng tiếng Việt. |
| `README.md` | Ghi chú ngắn về trạng thái kiểm thử của dự án. |
| `minimax_alpha_beta_vi.tex` | Báo cáo LaTeX tiếng Việt về Minimax và Alpha-Beta. |
| `minimax_alpha_beta_en.tex` | Phiên bản tiếng Anh của báo cáo Minimax và Alpha-Beta. |
| `uct_math_vi.tex` | Báo cáo LaTeX tiếng Việt về cơ sở toán học của UCT. |
| `uct_math_en.tex` | Phiên bản tiếng Anh của báo cáo UCT. |
| `*.pdf` | Bản PDF đã biên dịch từ các file LaTeX hoặc đề bài. |
| `*.aux`, `*.log`, `*.out`, `missfont.log` | File phụ do LaTeX tạo ra trong quá trình biên dịch. |
| `de thi bai tap lon AI 2026_2027.docx.pdf` | Đề bài gốc, gồm yêu cầu triển khai, thực nghiệm và báo cáo. |

Các file `aux`, `log`, `out` không chứa thuật toán mới. Chúng chỉ lưu thông tin
mục lục, nhãn, cảnh báo và log biên dịch; có thể tạo lại khi biên dịch LaTeX.

---

## 3. Mô hình trò chơi

### 3.1. Trạng thái

Với Caro, bàn cờ là ma trận `n × n`. Mỗi ô có một trong ba giá trị:

\[
\text{EMPTY} = '.',\qquad X,\qquad O.
\]

Một trạng thái có thể viết dưới dạng

\[
s=(B,p),
\]

trong đó `B` là bàn cờ và `p` là người sắp đi. Trong mã, `Board.cells` biểu
diễn `B`, còn `Board.player` biểu diễn `p`.

`X` là máy trong hai chương trình hiện tại. `O` là người chơi hoặc đối thủ ngẫu
nhiên tùy chương trình.

### 3.2. Nước đi và trạng thái kế tiếp

Tập nước đi hợp lệ của trạng thái `s` là

\[
A(s)=\{(r,c)\mid B[r][c]='.'\}.
\]

Hàm chuyển trạng thái `T(s,a)` đặt quân của người đang đi vào ô `a`, sau đó đổi
lượt:

\[
p'=
\begin{cases}
O,&p=X,\\
X,&p=O.
\end{cases}
\]

`Board.play()` không sửa trực tiếp bàn cờ hiện tại. Hàm tạo một bản sao, đặt
quân trên bản sao và trả về trạng thái mới. Cách này làm cây tìm kiếm dễ hiểu
và tránh việc một nhánh vô tình làm thay đổi nhánh khác, đổi lại là tốn thêm
thời gian và bộ nhớ cho việc sao chép ma trận.

### 3.3. Điều kiện thắng

Luật Caro trong mã là có ít nhất năm quân liên tiếp theo một trong bốn hướng:

\[
(0,1) \text{ ngang},\quad
(1,0) \text{ dọc},\quad
(1,1) \text{ chéo chính},\quad
(1,-1) \text{ chéo phụ}.
\]

Từ một quân tại `(r,c)`, hàm `winner()` đếm theo cả hai chiều của một hướng.
Nếu `k` là số quân liên tục thì trạng thái thắng khi

\[
k\ge 5.
\]

Đếm cả hai phía là cần thiết. Ví dụ, nếu bên trái có hai quân giống nhau và
bên phải có hai quân giống nhau, quân vừa đặt tạo thành chuỗi dài năm.

`Board.full()` trả về đúng khi không còn ô trống. Trạng thái kết thúc là

\[
\text{terminal}(s)=\text{winner}(s)\ne\varnothing
\;\lor\;\text{full}(s).
\]

### 3.4. Giảm số nước ứng viên

Trên bàn lớn, xét mọi ô trống làm hệ số phân nhánh rất lớn. `legal_moves()` dùng
chiến lược thực dụng:

- Nếu bàn rỗng, chỉ xét ô trung tâm.
- Nếu bàn đã có quân, chỉ xét các ô trống nằm trong bán kính Chebyshev 2 quanh
  ít nhất một quân đã đặt.
- Dùng `set` để loại bỏ các ô trùng nhau.

Đây là một **heuristic sinh nước**, không phải định nghĩa đầy đủ của luật. Nó
làm tìm kiếm nhanh hơn nhưng có thể loại bỏ một nước hợp lệ ở xa cụm quân. Vì
vậy, khi cần chứng minh tính tối ưu tuyệt đối, nên sinh toàn bộ ô trống hoặc
chứng minh miền ứng viên phù hợp với luật bài toán.

---

## 4. File `tic_tac_toe_ai.py`

### 4.1. Hằng số và kiểu dữ liệu

```python
EMPTY = "."
X = "X"
O = "O"
WIN_SCORE = 1_000_000
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
```

`WIN_SCORE` tạo khoảng cách rất lớn giữa trạng thái thắng và điểm heuristic.
`DIRECTIONS` được dùng chung cho kiểm tra thắng và quét các đoạn năm ô.

Lớp `Board` là `dataclass` với ba trường:

```python
size: int = 9
cells: list[list[str]]
player: str = X
```

`__post_init__()` tạo ma trận rỗng kích thước `size` nếu người gọi không truyền
`cells`.

### 4.2. `copy()`

```python
return Board(self.size, [row[:] for row in self.cells], self.player)
```

`row[:]` tạo bản sao từng hàng. Nếu chỉ sao chép tham chiếu tới các hàng, việc
đặt quân trong bản sao sẽ làm thay đổi bàn gốc. Đây là lỗi phổ biến khi biểu diễn
ma trận lồng nhau trong Python.

### 4.3. `play()`

Hàm kiểm tra ô đích còn trống; nếu không, phát sinh `ValueError`. Sau đó nó:

1. Sao chép bàn cờ.
2. Đặt `self.player` vào ô `(row,col)`.
3. Đổi `player` của trạng thái mới.
4. Trả về bàn cờ mới.

Hàm này bảo đảm một nước đi trong cây luôn tạo ra trạng thái mới hợp lệ.

### 4.4. `winner()`

Nếu có `last_move`, hàm chỉ kiểm tra từ nước vừa đặt. Đây là tối ưu quan trọng
trong lúc rollout: một nước mới chỉ có thể tạo chiến thắng ở các đường đi qua
ô mới. Nếu không truyền `last_move`, hàm duyệt tất cả quân đã có; cách này chậm
hơn nhưng thuận tiện cho `evaluate()` và các kiểm tra tổng quát.

Với mỗi hướng, thuật toán khởi tạo `count = 1`, rồi đi theo `sign = -1` và
`sign = 1`. Vòng lặp dừng khi ra ngoài bàn hoặc gặp ký hiệu khác.

### 4.5. `display()`

Hàm in chỉ số hàng và cột từ 0. Ví dụ bàn 5×5 được hiển thị để người dùng nhập
hoặc theo dõi nước dạng `(row, col)`. Chương trình hiện tại không có giao diện
đồ họa; giao diện là terminal.

---

## 5. Hàm lượng giá cho Minimax

### 5.1. `line_score(own, opponent)`

Mỗi cửa sổ dài năm được chấm như sau:

| Số quân của một bên | Điểm tuyệt đối |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 10 |
| 3 | 100 |
| 4 | 10 000 |
| 5 | 1 000 000 |

Nếu một cửa sổ có cả quân của `maximizer` và `minimizer`, điểm là 0 vì chuỗi
đã bị chặn. Nếu chỉ có quân của một bên, điểm có dấu tương ứng:

\[
L(o,d)=
\begin{cases}
0,&o>0\land d>0,\\
+w_o,&o>0\land d=0,\\
-w_d,&d>0\land o=0,\\
0,&o=d=0.
\end{cases}
\]

### 5.2. `evaluate(board, maximizer)`

Hàm trước hết kiểm tra người thắng. Nếu `maximizer` thắng, trả `+1 000 000`; nếu
đối thủ thắng, trả `-1 000 000`. Việc kiểm tra terminal trước heuristic ngăn một
trạng thái đã thắng bị đánh giá như trạng thái chưa kết thúc.

Sau đó hàm quét mọi ô làm điểm bắt đầu và bốn hướng. Chỉ giữ những đoạn mà ô
cuối `(row + 4*d_row, col + 4*d_col)` còn nằm trong bàn. Với mỗi đoạn, đếm quân
của hai bên rồi cộng `line_score`.

Đây là hàm xấp xỉ. Nó không biết đầy đủ chiến lược dài hạn, không tính trực tiếp
độ mở của hai đầu chuỗi và có thể đếm các đoạn chồng lấn. Các giá trị `1,10,100,
10 000` là lựa chọn thiết kế, không phải định lý tối ưu.

---

## 6. Minimax và Alpha-Beta trong mã Caro

### 6.1. Minimax

Gọi `V(s)` là giá trị của trạng thái khi hai bên đều chọn tốt nhất. Với người máy
là MAX:

\[
V(s)=
\begin{cases}
\operatorname{evaluate}(s),&s\text{ kết thúc hoặc hết độ sâu},\\
\max_{a\in A(s)}V(T(s,a)),&p=X,\\
\min_{a\in A(s)}V(T(s,a)),&p=O.
\end{cases}
\]

Ở nút gốc, `best_minimax_move()` thử từng nước, gọi `minimax()` cho bàn con và
giữ nước có giá trị lớn nhất nếu máy đang đi. Hàm cũng hỗ trợ trường hợp người
được truyền vào là bên MIN bằng cách lấy giá trị nhỏ nhất.

Tham số `depth` là số lớp còn được phép mở rộng. `depth = 1` nghĩa là đặt một
nước rồi đánh giá bàn con; `depth = 2` xét thêm một nước của đối thủ.

### 6.2. Cắt tỉa Alpha-Beta

`alpha` là cận dưới tốt nhất mà MAX đã đảm bảo trên đường hiện tại. `beta` là
cận trên tốt nhất mà MIN có thể ép trên đường đó. Sau khi cập nhật:

\[
\alpha\leftarrow\max(\alpha,v)\quad\text{ở MAX},
\]
\[
\beta\leftarrow\min(\beta,v)\quad\text{ở MIN}.
\]

Khi

\[
\alpha\ge\beta,
\]

các con còn lại không thể thay đổi giá trị mà tổ tiên cần. Mã tăng
`statistics["cutoffs"]` rồi `break` vòng lặp. Cắt tỉa không đổi kết quả Minimax;
nó chỉ giảm số nút phải đánh giá.

### 6.3. Độ phức tạp

Với hệ số phân nhánh `b` và độ sâu `d`, Minimax có cận thời gian

\[
O(b^d).
\]

Alpha-Beta có cùng cận xấu nhất, nhưng thứ tự nước tốt có thể đưa cận lý tưởng
xấp xỉ

\[
O(b^{d/2}).
\]

Trong mã này, `legal_moves()` làm giảm `b`; thứ tự từ `set` không được đảm bảo là
thứ tự chiến lược tốt. Có thể cải thiện bằng cách ưu tiên nước thắng ngay, nước
chặn thắng, ô trung tâm, hoặc nước gần chuỗi dài.

### 6.4. Số liệu theo dõi

- `nodes`: số lần gọi `minimax`, bao gồm cả các nút bị dừng ở terminal hoặc độ
  sâu bằng 0.
- `cutoffs`: số lần điều kiện `alpha >= beta` làm dừng vòng lặp.

Hai số này giúp so sánh Minimax thuần túy và Alpha-Beta, nhưng thời gian chạy còn
phụ thuộc Python, kích thước bàn, hàm lượng giá và cách sinh nước.

### 6.5. Animation: Minimax duyệt cây và Alpha–Beta cắt nhánh

Animation dưới đây đọc từ trái sang phải theo một chu kỳ: MAX mở rộng cây, MIN
đánh giá các con, rồi Alpha–Beta nhận ra rằng nhánh màu đỏ không thể cải thiện
đáp án đã có. Số cạnh được giữ lại cho thấy phần cây thực sự cần tính; dấu `×`
cho biết các nút phía dưới bị bỏ qua vì `alpha >= beta`.

<svg viewBox="0 0 760 300" role="img" aria-labelledby="minimax-animation-title minimax-animation-desc" style="width:100%;height:auto;max-width:760px">
  <title id="minimax-animation-title">Minimax và Alpha-Beta pruning</title>
  <desc id="minimax-animation-desc">Cây tìm kiếm gồm nút MAX, hai nút MIN và các giá trị lá. Nhánh đỏ bị cắt sau khi cận alpha lớn hơn hoặc bằng cận beta.</desc>
  <style>
    .mm-line{stroke:currentColor;stroke-width:2;opacity:.35}
    .mm-node{stroke:currentColor;stroke-width:2}
    .mm-label{font:500 14px sans-serif;fill:currentColor;text-anchor:middle}
    .mm-small{font:400 12px sans-serif;fill:currentColor}
    .mm-a{fill:#4f8cff}.mm-b{fill:#f3a641}.mm-leaf{fill:#8bc98b}.mm-cut{fill:#e96b6b}
    @media (prefers-reduced-motion: reduce){.mm-animated{animation:none!important}}
  </style>
  <g aria-hidden="true">
    <path class="mm-line" d="M380 48L190 126M380 48L570 126"/>
    <path class="mm-line" d="M190 146L100 226M190 146L280 226M570 146L480 226M570 146L660 226"/>
    <path class="mm-line mm-animated" d="M570 146L660 226" stroke="#e96b6b" opacity="0">
      <animate attributeName="opacity" values="0;0;1;1;0" dur="8s" repeatCount="indefinite"/>
    </path>
    <circle class="mm-node mm-a" cx="380" cy="38" r="25"/>
    <circle class="mm-node mm-b" cx="190" cy="136" r="23"/>
    <circle class="mm-node mm-b" cx="570" cy="136" r="23"/>
    <circle class="mm-node mm-leaf" cx="100" cy="236" r="20"/>
    <circle class="mm-node mm-leaf" cx="280" cy="236" r="20"/>
    <circle class="mm-node mm-leaf" cx="480" cy="236" r="20"/>
    <circle class="mm-node mm-cut" cx="660" cy="236" r="20" opacity=".35"/>
    <text class="mm-label" x="380" y="43">MAX</text>
    <text class="mm-label" x="190" y="141">MIN</text><text class="mm-label" x="570" y="141">MIN</text>
    <text class="mm-label" x="100" y="241">3</text><text class="mm-label" x="280" y="241">5</text>
    <text class="mm-label" x="480" y="241">2</text><text class="mm-label" x="660" y="241">9</text>
    <text class="mm-small" x="18" y="24">1 · MAX chọn nhánh đầu</text>
    <text class="mm-small" x="18" y="284">2 · MIN trả min(3, 5) = 3</text>
    <text class="mm-small" x="470" y="284">3 · nhánh đỏ bị cắt</text>
    <text class="mm-label mm-animated" x="660" y="194" fill="#e96b6b" opacity="0">
      <tspan>α ≥ β → cắt</tspan>
      <animate attributeName="opacity" values="0;0;1;1;0" dur="8s" repeatCount="indefinite"/>
    </text>
    <text class="mm-label" x="570" y="174" opacity=".75">β = 2</text>
    <text class="mm-label" x="380" y="82" opacity=".75">α = 3</text>
  </g>
</svg>

Mỗi vòng lặp animation tương ứng với cách đọc thuật toán:

1. Ở nút `MAX`, ta cần giá trị lớn nhất giữa các nước con. Ở nút `MIN`, ta cần
   giá trị nhỏ nhất; vì vậy nhánh trái trả `min(3,5)=3` và cập nhật `alpha=3`.
2. Khi xét nút `MIN` bên phải, chỉ cần thấy một con có giá trị `2` là nút này
   sẽ không thể trả quá `2` cho MAX, nên `beta=2`.
3. Vì `alpha=3 >= beta=2`, lá `9` và mọi hậu duệ còn lại không thể làm thay đổi
   lựa chọn của MAX. Cắt chúng không làm đổi đáp án Minimax, chỉ giảm số lần gọi
   hàm và số phép lượng giá.

---

## 7. MCTS và UCT trong mã Caro

### 7.1. Cấu trúc `MCTSNode`

Mỗi nút lưu:

| Trường | Ý nghĩa |
|---|---|
| `board` | Trạng thái bàn cờ tại nút. |
| `parent` | Nút cha. |
| `move` | Nước đi từ cha đến nút này. |
| `player_just_moved` | Người vừa đặt quân để quy đổi kết quả. |
| `children` | Các con đã được mở rộng. |
| `untried_moves` | Nước hợp lệ chưa tạo thành con. |
| `visits` | Số simulation đi qua nút. |
| `wins` | Tổng phần thưởng tích lũy. |

Trong `__post_init__`, danh sách `untried_moves` được khởi tạo từ trạng thái bàn.

### 7.2. Công thức UCT

Với nút con `i` của nút cha `p`, mã dùng

\[
\operatorname{UCT}(i)=
\frac{W_i}{N_i}
 +c\sqrt{\frac{\ln N_p}{N_i}}.
\]

Trong đó:

- `W_i` là tổng điểm thắng của nút con.
- `N_i` là số lần nút con được thăm.
- `N_p` là số lần nút cha được thăm.
- `c` là hằng số cân bằng khai thác và khám phá.

Nếu `N_i=0`, `uct()` trả `+∞`. Vì thế mọi con mới đều được ưu tiên ít nhất một
lần và không bị chia cho 0.

Thành phần đầu là **khai thác**: ưu tiên nước có trung bình tốt. Thành phần thứ
hai là **khám phá**: ưu tiên nước ít được thử. Khi `N_i` tăng, bán kính không chắc
chắn giảm theo `1/\sqrt{N_i}`; khi `N_p` tăng, logarit làm các nhánh bị bỏ quên
tiếp tục có cơ hội được thử.

Giá trị thường dùng là

\[
c=\sqrt 2\approx1.414,
\]

nhưng đây chỉ là điểm bắt đầu. Nên thực nghiệm các giá trị `0.5, 1.0, 1.4, 2.0`.

### 7.3. Vì sao xuất hiện căn và logarit?

Xem mỗi nước đi như một arm trong bài toán Multi-Armed Bandit. Gọi `\hat\mu_i`
là phần thưởng trung bình quan sát được và `\mu_i` là kỳ vọng thật. Với phần
thưởng trong `[0,1]`, bất đẳng thức Hoeffding cho:

\[
P(|\hat\mu_i-\mu_i|\ge\varepsilon)
\le 2e^{-2N_i\varepsilon^2}.
\]

Chọn bán kính sai số tỷ lệ với

\[
\varepsilon_i=\sqrt{\frac{2\ln N_p}{N_i}}
\]

cho một cận tin cậy tăng chậm theo số lần thăm cha và giảm theo số mẫu của con.
Do đó UCB/UCT có dạng trung bình thực nghiệm cộng bán kính tin cậy. Trong thực
tế, phần thưởng trong cây trò chơi phụ thuộc chính sách ở các nút dưới và mẫu
được chọn thích nghi, nên đây là cơ sở lý thuyết và trực giác, không phải giả
định độc lập hoàn hảo cho toàn bộ cây.

### 7.4. Bốn pha của MCTS

Mỗi vòng trong `best_mcts_move()` thực hiện:

1. **Selection**: từ root, chọn con có UCT lớn nhất khi nút đã mở rộng hết và
   chưa kết thúc.
2. **Expansion**: lấy một nước từ `untried_moves`, tạo `MCTSNode` mới và nối
   vào `children`.
3. **Simulation/Rollout**: từ nút mới, chơi tiếp đến khi thắng hoặc bàn đầy.
4. **Backpropagation**: đi ngược qua `parent`, tăng `visits` và cộng phần thưởng
   vào `wins`.

Sau đủ vòng, chương trình chọn con của root có `visits` lớn nhất. Chọn nhiều lượt
thăm thường ổn định hơn chọn `wins/visits` lớn nhất dựa trên chỉ một vài mẫu.

### 7.5. Quy ước phần thưởng

Trong backpropagation hiện tại:

\[
z=
\begin{cases}
0.5,&\text{hòa},\\
1,&\text{người vừa đi thắng},\\
0,&\text{người vừa đi thua}.
\end{cases}
\]

`player_just_moved` giúp mỗi nút nhìn kết quả theo người đã tạo ra nút đó. Đây là
quy ước hợp lệ, nhưng phải nhất quán với cách chọn UCT. Khi thay đổi sang phần
thưởng nhìn từ root, cần sửa lại backpropagation và cách diễn giải `wins`.

### 7.6. Animation: một vòng MCTS–UCT

Animation này minh họa một iteration hoàn chỉnh. Vòng sáng chạy qua `Selection`,
`Expansion`, `Simulation` và `Backpropagation`; sau mỗi vòng, một nhánh mới được
thêm và số lượt thăm được đẩy ngược về root. UCT vì thế vừa ưu tiên nhánh có
`wins/visits` cao, vừa dành cơ hội cho nhánh ít được thăm.

<svg viewBox="0 0 760 330" role="img" aria-labelledby="mcts-animation-title mcts-animation-desc" style="width:100%;height:auto;max-width:760px">
  <title id="mcts-animation-title">Bốn pha của MCTS với UCT</title>
  <desc id="mcts-animation-desc">Một nút gốc có các nhánh con. Animation lần lượt chọn nhánh, mở rộng nút mới, mô phỏng đến lá và truyền kết quả ngược về gốc.</desc>
  <style>
    .mc-line{stroke:currentColor;stroke-width:2;opacity:.35;fill:none}
    .mc-node{stroke:currentColor;stroke-width:2}.mc-root{fill:#4f8cff}.mc-old{fill:#f3a641}.mc-new{fill:#8bc98b}.mc-roll{fill:#b69ae8}
    .mc-label{font:500 14px sans-serif;fill:currentColor;text-anchor:middle}.mc-small{font:400 12px sans-serif;fill:currentColor}
    @media (prefers-reduced-motion: reduce){.mc-animated{animation:none!important}}
  </style>
  <g aria-hidden="true">
    <path class="mc-line" d="M380 70L230 150M380 70L530 150M230 170L155 250M230 170L305 250M530 170L605 250"/>
    <path class="mc-line mc-animated" d="M530 170L605 250" stroke="#8bc98b" stroke-width="5" opacity="0">
      <animate attributeName="opacity" values="0;0;1;1;0" dur="8s" repeatCount="indefinite"/>
    </path>
    <circle class="mc-node mc-root" cx="380" cy="60" r="27"/><circle class="mc-node mc-old" cx="230" cy="160" r="23"/><circle class="mc-node mc-old" cx="530" cy="160" r="23"/>
    <circle class="mc-node mc-roll" cx="155" cy="260" r="20"/><circle class="mc-node mc-new" cx="305" cy="260" r="20"/><circle class="mc-node mc-new" cx="605" cy="260" r="20"/>
    <text class="mc-label" x="380" y="65">R: 12</text><text class="mc-label" x="230" y="165">A: 8</text><text class="mc-label" x="530" y="165">B: 4</text>
    <text class="mc-label" x="155" y="265">L</text><text class="mc-label" x="305" y="265">L</text><text class="mc-label" x="605" y="265">N</text>
    <text class="mc-small" x="22" y="24">Selection → UCT chọn A/B</text><text class="mc-small" x="22" y="310">Simulation: rollout trả thưởng z</text><text class="mc-small" x="512" y="310">Backprop: cập nhật R ← R+z</text>
    <text class="mc-label mc-animated" x="380" y="112" fill="#8bc98b" opacity="0">UCT cao nhất
      <animate attributeName="opacity" values="0;1;1;0;0" dur="8s" repeatCount="indefinite"/>
    </text>
    <circle class="mc-animated" cx="530" cy="160" r="31" fill="none" stroke="#8bc98b" stroke-width="4" opacity="0">
      <animate attributeName="opacity" values="0;0;1;1;0" dur="8s" repeatCount="indefinite"/>
    </circle>
    <path class="mc-animated" d="M605 250L530 175L380 75" stroke="#8bc98b" stroke-width="4" fill="none" opacity="0" stroke-dasharray="8 6">
      <animate attributeName="opacity" values="0;0;0;1;0" dur="8s" repeatCount="indefinite"/>
    </path>
  </g>
</svg>

Khi `N_i=0`, nhánh mới được xem như có UCT `+∞`, nên luôn được thử một lần. Sau
đó, mỗi simulation cập nhật `visits` và `wins` trên đường đi từ lá về root. Sau
`M` iteration, `best_mcts_move()` chọn con root có nhiều lượt thăm nhất; đây là
ước lượng từ mẫu, khác với Minimax vốn tính giá trị bằng cách xét đối thủ theo
kịch bản xấu nhất.

---

## 8. Rollout ngẫu nhiên và rollout heuristic

### 8.1. Rollout ngẫu nhiên

Nếu `heuristic=False`, mỗi bước chọn ngẫu nhiên một phần tử của `legal_moves()`.
Rollout dừng khi `current.terminal()` đúng. Kết quả trả về là `X`, `O` hoặc `None`
cho hòa.

Rollout ngẫu nhiên đơn giản và ít thiên lệch theo chiến lược, nhưng có thể rất
yếu: nó thường bỏ qua nước thắng ngay hoặc không chặn đối thủ.

### 8.2. Rollout heuristic hiện có

Nếu `heuristic=True`, chương trình trước hết tìm nước thắng ngay cho người đang
đi. Nếu không có, mã cố tìm nước chặn đối thủ rồi mới chọn ngẫu nhiên.

Ý tưởng này hợp lý về mặt AI, nhưng cần chú ý một chi tiết của phiên bản hiện
tại: trong vòng tìm `blocks`, mã tạo `candidate = current.play(move)`, mà nước
đó do `current.player` đặt. Vì vậy điều kiện

```python
candidate.winner(move) == opponent
```

thường không thể đúng: `candidate.winner(move)` kiểm tra ký hiệu của người vừa
đặt, không phải đối thủ. Do đó nhánh chặn đối thủ hiện có thể không bao giờ được
chọn. Có thể sửa bằng cách tạo một hàm kiểm tra giả định đối thủ đặt vào ô đó,
hoặc kiểm tra trạng thái sau khi cho đối thủ chơi nước tương ứng. Ví dụ về ý
tưởng:

```python
def would_win(board, move, player):
    test = board.copy()
    test.cells[move[0]][move[1]] = player
    return test.winner(move) == player
```

Sau đó chọn các `move` sao cho `would_win(current, move, opponent)` đúng. Khi
sửa, nên chạy lại thực nghiệm vì rollout heuristic đã thay đổi phân phối mẫu.

---

## 9. Luồng chạy một ván Caro

`play_game(args)` khởi tạo bàn, rồi lặp khi chưa kết thúc:

1. In bàn và người đang đi.
2. Đo thời gian bằng `time.perf_counter()`.
3. Nếu là `X`, gọi Minimax hoặc MCTS theo `--mode`.
4. Nếu là `O`, chọn ngẫu nhiên bằng `random_move()`.
5. In nước đi và thời gian.
6. Gọi `board.play(move)` để sang trạng thái mới.
7. Khi kết thúc, in bàn cuối và người thắng hoặc `Hòa`.

Mặc định chỉ máy `X` dùng AI; `O` là đối thủ ngẫu nhiên. Vì vậy kết quả của một
ván không phải là so sánh trực tiếp giữa hai thuật toán nếu chỉ đổi `--mode`:
đó là so sánh với cùng kiểu đối thủ ngẫu nhiên, trong một seed cụ thể.

---

## 10. Tham số dòng lệnh

```text
--mode          minimax hoặc mcts; mặc định mcts
--size          kích thước bàn; mặc định 9, phải >= 5
--depth         độ sâu Minimax; mặc định 2
--iterations    số vòng MCTS; mặc định 1000
--exploration   c trong UCT; mặc định sqrt(2)
--heuristic     bật rollout heuristic
--seed          seed ngẫu nhiên; mặc định 7
```

Ví dụ:

```bash
python tic_tac_toe_ai.py --mode mcts --size 9 --iterations 1000
python tic_tac_toe_ai.py --mode mcts --size 9 --iterations 5000 --exploration 1.4 --heuristic --seed 12
python tic_tac_toe_ai.py --mode minimax --size 5 --depth 2
```

Chương trình kiểm tra `size < 5` và dừng bằng lỗi đối số. Minimax trên bàn lớn
với độ sâu cao có thể tăng thời gian rất nhanh do bùng nổ số nhánh.

---

## 11. File `ttt_minimax.py`: Tic-Tac-Toe 3×3 chính xác

Đây là chương trình nhỏ hơn, dùng chuỗi chín ký tự để biểu diễn bàn:

```text
.........
```

Ký tự `.` là ô trống. Chỉ số chuỗi từ 0 đến 8 được ánh xạ thành bàn:

```text
0 1 2
3 4 5
6 7 8
```

`WIN_LINES` chứa tám đường thắng: ba hàng, ba cột và hai đường chéo.

- `winner(s)`: trả `X`, `O` hoặc `None`.
- `terminal(s)`: đúng nếu có người thắng hoặc không còn `.`.
- `utility(s, root)`: thắng từ góc nhìn `root` là `+1`, thua `-1`, hòa `0`.
- `moves(s)`: sinh chỉ số các ô trống.
- `put(s,i,p)`: tạo chuỗi mới sau khi đặt quân.

### 11.1. Minimax 3×3

Không dùng độ sâu giới hạn. Hàm tiếp tục đến terminal, vì cây Tic-Tac-Toe hữu
hạn và đủ nhỏ. Điều đó làm giá trị trả về là lời giải chính xác dưới giả thiết
hai bên chơi tối ưu.

### 11.2. Alpha-Beta 3×3

Hàm `alphabeta()` giống Minimax nhưng nhận `alpha` và `beta`, khởi tạo ở root là
`-1` và `1` vì utility chỉ nằm trong `[-1,1]`. `--show-search` in giá trị của
từng ô ứng viên, còn chương trình luôn in số nút và số lần cắt.

Một điểm kỹ thuật: `from functools import lru_cache` được import nhưng chưa sử
dụng. Vì thế chương trình hiện không có memoization/transposition table. Có thể
dùng `@lru_cache` nếu chuyển các hàm tìm kiếm sang dạng phù hợp, nhưng phải đưa
đầy đủ trạng thái người đang đi vào khóa cache.

Chạy:

```bash
python ttt_minimax.py --algorithm alphabeta --show-search
python ttt_minimax.py --algorithm minimax
```

Máy là `X`, người nhập nước là `O`. Người dùng nhập số từ 1 đến 9; chương trình
trừ 1 để chuyển sang chỉ số nội bộ. Nếu nhập ô đã dùng, chương trình báo lỗi và
giữ nguyên lượt. Một input không phải số nguyên hiện sẽ phát sinh `ValueError`.

---

## 12. So sánh hai thuật toán

| Đặc điểm | Minimax–Alpha-Beta | MCTS–UCT |
|---|---|---|
| Kiểu tìm kiếm | Có hệ thống, duyệt cây đến độ sâu đặt trước hoặc terminal | Dựa trên mẫu mô phỏng, mở rộng dần cây |
| Đầu vào quan trọng | Độ sâu, hàm lượng giá, thứ tự nước | Số iteration, `c`, rollout, phần thưởng |
| Tính tối ưu | Chính xác nếu duyệt hết cây và sinh đủ nước | Hội tụ tiệm cận dưới giả thiết; hữu hạn thì xấp xỉ |
| Phù hợp | Tic-Tac-Toe nhỏ, trạng thái có thể duyệt sâu | Caro/bàn lớn, cây quá rộng |
| Điểm yếu | Bùng nổ theo `b^d`, phụ thuộc heuristic khi cắt độ sâu | Có thể cần rất nhiều rollout, nhạy với rollout và `c` |
| Dữ liệu đo | `nodes`, `cutoffs`, thời gian, giá trị nước | `root.visits`, số con, thời gian, thắng/hòa/thua |

Alpha-Beta chỉ thực sự hiệu quả khi nước tốt được xét sớm. MCTS không cần mở
rộng toàn bộ các nhánh, nhưng chất lượng phụ thuộc vào việc các mô phỏng phản ánh
được chiến lược thật.

---

## 13. Thiết kế thực nghiệm đúng

Một thực nghiệm có thể dùng các cấu hình:

\[
c\in\{0.5,1.0,1.4,2.0\},
\qquad
M\in\{100,500,1000,5000\}.
\]

Với mỗi cấu hình nên chơi ít nhất 30 ván, dùng nhiều seed và đổi người đi trước.
Nên ghi:

- tỷ lệ thắng, hòa, thua;
- thời gian trung bình mỗi nước và mỗi ván;
- số nút Minimax và số cutoff;
- số lượt thăm root và số con root của MCTS;
- kích thước bàn, luật thắng và người đi trước;
- kiểu rollout: ngẫu nhiên hay heuristic;
- đối thủ: ngẫu nhiên, Minimax, MCTS hoặc người chơi tối ưu 3×3.

Tỷ lệ thắng của một cấu hình là

\[
\widehat p_{win}=\frac{W}{G},\qquad
\widehat p_{draw}=\frac{D}{G},\qquad
\widehat p_{loss}=\frac{L}{G},
\]

với `G=W+D+L`. Ba tỷ lệ phải cộng bằng 1, ngoại trừ sai số làm tròn.

Không nên kết luận `c=1.4` luôn tốt nhất chỉ từ một ván. Một giá trị tốt phụ
thuộc bàn cờ, rollout, ngân sách, đối thủ và thứ tự đi. Với Tic-Tac-Toe 3×3,
có thể dùng `ttt_minimax.py` làm đối thủ tối ưu để đánh giá MCTS đáng tin cậy hơn.

---

## 14. Nội dung các báo cáo LaTeX và PDF

### 14.1. Báo cáo Minimax

`minimax_alpha_beta_vi.tex` và bản tiếng Anh trình bày:

- mô hình trạng thái và giả thiết trò chơi tổng bằng không;
- định nghĩa đệ quy Minimax;
- trực giác chứng minh tính đúng bằng quy nạp theo độ sâu;
- ý nghĩa của `alpha`, `beta` và điều kiện cắt;
- phân tích `O(b^d)` và trường hợp tốt xấp xỉ `O(b^{d/2})`;
- tác động của thứ tự nước;
- cách theo dõi `nodes`, `cutoffs`;
- mã `ttt_minimax.py` bằng `lstinputlisting`;
- tài liệu tham khảo Russell–Norvig, Knuth–Moore và Shannon.

### 14.2. Báo cáo UCT

`uct_math_vi.tex` và bản tiếng Anh trình bày:

- mô hình trò chơi hữu hạn `G=(S,A,T,s0,Z,r)`;
- bài toán Multi-Armed Bandit;
- trung bình mẫu, regret và cận Hoeffding;
- suy ra UCB1 rồi mở rộng thành UCT trên cây;
- bốn pha Selection, Expansion, Simulation, Backpropagation;
- tính nhất quán tiệm cận và giới hạn của ngân sách hữu hạn;
- ánh xạ công thức vào `MCTSNode` và `best_mcts_move`;
- rollout ngẫu nhiên và heuristic;
- thiết kế thực nghiệm với các giá trị `c` và iteration;
- tài liệu tham khảo Auer, Kocsis–Szepesvári, Browne và Sutton–Barto.

Biên dịch các file LaTeX bằng XeLaTeX, ví dụ:

```bash
xelatex minimax_alpha_beta_vi.tex
xelatex uct_math_vi.tex
```

Thường cần biên dịch hai lần nếu có mục lục hoặc tham chiếu chéo. Các file phụ
được tạo ra chính là `aux`, `log` và `out`.

### 14.3. `DAP_AN_AI.md` và `README.md`

`DAP_AN_AI.md` là khung trả lời tiếng Việt, gồm phát biểu bài toán, công thức,
pseudocode, diễn giải UCT, ánh xạ vào code và đề xuất bảng thực nghiệm. Nó nhắc
đúng rằng số liệu phải được điền sau khi chạy thực tế.

`README.md` hiện chỉ có ghi chú rất ngắn rằng đây là code và báo cáo thử nghiệm,
đồng thời nói phiên bản chưa được xác minh đầy đủ. File Markdown này bổ sung
phần giải thích chi tiết còn thiếu.

---

## 15. Những giới hạn và việc nên cải thiện

1. **Cửa sổ ứng viên quanh quân** có thể loại bỏ nước xa. Nếu mục tiêu là chứng
   minh tối ưu, cần sinh toàn bộ ô trống hoặc ghi rõ đây là heuristic.
2. **Rollout chặn đối thủ** trong phiên bản hiện tại cần sửa như phân tích ở
   mục 8.2.
3. `random_move()` tạo `Random(seed)` ở mỗi lượt. Với cùng seed, trình tự có
   thể lặp lại theo cách không mong muốn; nên tạo một RNG dùng chung cho cả ván.
4. MCTS không tái sử dụng cây giữa các lượt. Sau khi chọn một nước, cây cũ bị bỏ;
   giữ lại cây con tương ứng sẽ tiết kiệm nhiều simulation.
5. Chưa có bảng chuyển trạng thái, nên các trạng thái giống nhau được tính lại.
6. Hàm lượng giá chưa phân biệt chuỗi mở một đầu, mở hai đầu, nước tạo fork
   hoặc mối đe dọa bắt buộc.
7. Không có kiểm thử tự động cho `winner`, `play`, `legal_moves`, điểm lượng giá
   và các trường hợp biên.
8. `ttt_minimax.py` import `lru_cache` nhưng chưa dùng và chưa bắt lỗi input
   không phải số.
9. Các báo cáo vẫn có placeholder như tên sinh viên/trường và chưa tự động chèn
   bảng số liệu thực nghiệm.
10. Các PDF hiện có là artifact đã biên dịch; muốn chắc chắn nội dung phản ánh
    mã mới nhất cần biên dịch lại sau khi chỉnh `.tex`.

---

## 16. Checklist để hoàn thiện bài nộp

- [ ] Chạy cả hai chương trình với các tham số nhỏ để kiểm tra không lỗi.
- [ ] Kiểm tra đủ tám đường thắng của Tic-Tac-Toe.
- [ ] Kiểm tra bốn hướng và chuỗi từ năm quân của Caro.
- [ ] Quyết định rõ dùng toàn bộ nước hay cửa sổ quanh quân trong báo cáo.
- [ ] Sửa và kiểm thử rollout chặn đối thủ nếu dùng chế độ heuristic.
- [ ] Đo ít nhất 50 ván cho mỗi cấu hình theo yêu cầu đề bài.
- [ ] Đổi người đi trước và dùng nhiều seed.
- [ ] Ghi bảng thắng/hòa/thua, thời gian, node, cutoff, `W`, `N`, `W/N`.
- [ ] Vẽ đồ thị iteration–tỷ lệ thắng cho từng `c`.
- [ ] Không đưa số liệu chưa đo vào phần tóm tắt hay kết luận.
- [ ] Điền tên, trường, môi trường chạy và phiên bản Python.
- [ ] Biên dịch lại PDF và kiểm tra các công thức, code listing, tài liệu tham khảo.

## 17. Kết luận

Dự án minh họa hai cách ra quyết định trong trò chơi đối kháng. Minimax mô hình
hóa trực tiếp giả thiết đối thủ luôn phản hồi tốt nhất; Alpha-Beta giữ nguyên
đáp án nhưng giảm công việc nhờ cận. MCTS–UCT dành ngân sách có giới hạn cho
các nhánh hứa hẹn và các nhánh chưa đủ dữ liệu; nó phù hợp với cây Caro lớn,
nhưng kết quả hữu hạn phụ thuộc mạnh vào số simulation, hằng số `c`, cách sinh
nước, quy ước phần thưởng và rollout.

Tic-Tac-Toe 3×3 là môi trường kiểm chứng chính xác vì có thể giải trọn cây.
Caro 5 quân là môi trường thực tế hơn để quan sát đánh đổi giữa độ sâu tìm kiếm,
heuristic và ngân sách mô phỏng. Khi báo cáo tách bạch kết quả lý thuyết với số
đo thực nghiệm, bài làm sẽ vừa giải thích được thuật toán vừa có thể kiểm chứng
lại bằng cách chạy mã.
