# TCOM-TPS-26-1667: MAJOR REVISION (thư 18/09/2026), phân tích và kế hoạch sửa

Bài: *KAN-Based Adaptive Parameter Control for Multi-User Satellite FSO/QKD Systems* (P. H. Do, M. Q. Vu, N. T. Dang).
Editor: Dr. Neel Kanth Kundu. Nộp 12/06/2026, quyết định sau 98 ngày. **Hạn nộp bản sửa: 60 ngày, tức 17/11/2026**
(hệ thống "strictly enforced"). Nguồn nhận xét: https://justpaste.it/gp5vm (chép lại ngày 19/09).
Yêu cầu nộp: bản sửa + *detailed response to all received comments* + mô tả đã sửa ở đâu.

## 1. Đọc tổng thể: không ai đụng scope, không ai đụng mô hình vật lý

Bốn phản biện, bốn giọng khác nhau, nhưng gộp lại thành **sáu gói việc**, trong đó ba gói nặng đều xoay quanh
cùng một sự thật mà chính số liệu của bài đã lộ: **KAN thắng MLP rất mỏng và phương sai theo seed che mất khác biệt.**

Số thật trong `results/` (bản đã nộp):
- Bảng IV (cụm, 200/50 cụm, **5 seed**): keyRet KAN **0.612 ± 0.484**, MLP **0.658 ± 0.400**, Linear 0.562 ± 0.000.
  Độ lệch chuẩn bằng cỡ giá trị; MLP còn cao hơn KAN. R2 chỉ đúng chỗ này.
- Bảng III (một liên kết, 198/50 trạng thái, 5 seed): keyRet KAN 0.716 ± 0.123, MLP 0.683 ± 0.086; KAN suy diễn **116,7 µs**
  so với MLP **5,0 µs**; KAN 392 tham số so với MLP 1282; KAN cho công thức tượng trưng của β*.
- Bộ dữ liệu ~250 mẫu (R4.8 đếm đúng).

Điểm sáng được nhận: mô phỏng TLE thật, bài toán tối ưu chung, trình bày rõ (R1, R2, R3, R4 đều ghi).

## 2. Mười chín nhận xét gộp thành sáu gói việc

| Gói | Nhận xét | Việc phải làm | Loại |
|---|---|---|---|
| **A. Thống kê và cỡ mẫu** | R1.4, R2.2, R4.7, R4.8 | (1) tăng seed 5 → ít nhất 20 cho Bảng III, IV; (2) báo **hiệu giữa cặp theo seed** (KAN trừ MLP cùng seed) kèm khoảng tin cậy bootstrap và kiểm định ghép cặp (Wilcoxon), thay cho mean ± sd; (3) nêu rõ mean hay median; (4) tăng số cụm oracle 250 → 1000 nếu solver kham nổi (đo thời gian `optimize_pass_cluster.py` trước); nếu không, vẽ **đường học** keyRet theo cỡ tập huấn luyện (50/100/200/…) để trả lời "dataset có đủ không" bằng số | thực nghiệm, chạy máy chủ |
| **B. Vì sao KAN, và KAN khác MLP ở đâu** | R1.1, R2.3, R3.4 | Viết lại §motivation trung thực: KAN không thắng về keyRet; KAN thắng ở (a) **tính diễn giải**: công thức β* dạng đóng (Eq. 21) MLP không cho được; (b) **ít tham số** 392 vs 1282; (c) **hiệu quả dữ liệu** (đường học của gói A sẽ cho thấy, nếu đúng); KAN thua ở thời gian suy diễn 23 lần. Thêm bảng "chọn gì khi nào". Giải thích vì sao so với MLP/KNN/Linear chứ không so với scheme khác | viết + một thí nghiệm |
| **C. Bảo đảm lý thuyết** | R1.2, R4.3 | Bài toán (P) không lồi; solver phân rã là block-coordinate. Có thể chứng minh được và trung thực: (i) mục tiêu bị chặn và **đơn điệu không giảm qua mỗi khối** ⇒ dãy giá trị hội tụ; (ii) điểm dừng là điểm dừng theo khối (block-stationary) dưới điều kiện tính liên tục; (iii) **không** hứa tối ưu toàn cục, thay bằng **khoảng cách tới lưới tìm kiếm toàn cục** đo được trên tập nhỏ (optimality gap). Với KAN: hội tụ huấn luyện chỉ nói theo lý thuyết xấp xỉ (Kolmogorov–Arnold) + bằng chứng thực nghiệm, ghi rõ giới hạn | toán + một thí nghiệm nhỏ |
| **D. Bảo mật** | R2.1, R3.1, R4.2 | (1) τ = 0,1: nêu nguồn (ngưỡng rò rỉ liên người dùng), **quét τ ∈ {0,02; 0,05; 0,1; 0,2}** và vẽ hệ quả lên khoá và tính khả thi; (2) mô tả BSA đầy đủ: kênh của Eve, hệ số truyền của bộ chia, vị trí, giả định; (3) trả lời thẳng "URA + BSA có đủ không": nêu đây là hai mô hình chuẩn của dòng bài SIM-BPSK/DT-DD (vu2023photonics và các bài gốc), liệt kê chiến lược khác (intercept-resend, thu thập photon tối ưu) và **vì sao ngoài phạm vi** hoặc thêm một chiến lược nếu rẻ; (4) "formal security": nối chỉ số P_e^E > 0,1 với cận tốc độ bí mật kiểu wiretap có trích dẫn, và nói rõ đây là chỉ số bảo mật ở lớp vật lý chứ không phải chứng minh composable | viết + quét tham số |
| **E. Thực tế và baseline** | R1.3, R4.4, R4.5 | (1) mở rộng điều kiện: mức nhiễu loạn/thời tiết, quỹ đạo khác, khoảng cách Eve; (2) thời gian triển khai: suy diễn 117 µs so với động lực kênh theo giây trong một lượt qua ⇒ dư nhiều bậc; nói cả thời gian huấn luyện và cập nhật lại; (3) **so với học tăng cường**: hai lựa chọn, (a) thêm một baseline RL nhỏ (một liên kết, ví dụ DDPG/PPO trên môi trường giải tích) và so keyRet + số tương tác cần; (b) chỉ thảo luận. Khuyến nghị (a) ở mức tối thiểu vì R4 hỏi thẳng "compare" | thực nghiệm |
| **F. Trình bày** | R3.2, R3.3, R3.5, R4.1, R4.6 | Bảng ký hiệu; sửa Hình 4 (chú thích, cỡ chữ, vị trí); "2.10×" → "2.10 times"; sửa chính tả và ngữ pháp toàn bài; **[9] thiếu tên tác giả** (kiểm cả bib); thêm bài tổng quan QKD vệ tinh và tài liệu 2024–2026 | biên tập |

Thứ tự làm: **A trước** (mọi kết luận về KAN vs MLP phụ thuộc vào nó; nếu 20 seed cho thấy MLP thật sự ngang KAN thì
gói B viết theo hướng đó, không ngược lại), rồi C và D song song, E, cuối cùng F. Không viết câu kết luận nào trước khi có số (lớp lỗi 29).

## 3. Rủi ro và điều không nên làm

- Không "đánh bóng" KAN: R1 và R2 đã nhìn thấy khoảng cách mỏng; trả lời trung thực + đường học + diễn giải được là con đường duy nhất.
- Không thêm chứng minh hội tụ toàn cục cho bài toán không lồi; R1/R4 sẽ bắt ngay.
- Baseline RL nếu làm cẩu thả (không tune) sẽ thành điểm yếu mới; nếu làm thì ghi rõ ngân sách tương tác và seed.
- Lịch: hạn 17/11 trùng AICON 20/11 và 27/11 DAU. Mốc đề nghị: số liệu gói A + E xong **20/10**; bản viết đủ **05/11**; đọc ngoài **08/11**; nộp **12–14/11** (không nộp sát hạn, không nộp quá sớm).

## 4. Việc phải xác nhận trước khi chạy

1. Máy chủ GPU còn dùng được cho `train_kan.py --seeds 20` và `train_cluster.py` (thời gian mỗi seed?).
2. Chi phí oracle cho 1000 cụm (`optimize_pass_cluster.py` in `elapsed`).
3. Chia việc với M. Q. Vu và N. T. Dang (gói D bảo mật hợp với người làm QKD; gói C toán).
