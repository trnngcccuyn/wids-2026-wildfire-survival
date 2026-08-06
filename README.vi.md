# Dự báo nguy cơ cháy rừng đe dọa vùng sơ tán — WiDS Global Datathon 2026

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-WiDS%202026-20BEFF.svg)](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)

> **Kết quả: 210 / 1.754 đội — top 12%**
> Mô hình phân tích sống sót (survival analysis) có kiểm duyệt phải, dự báo tốc độ một
> đám cháy rừng đe dọa vùng sơ tán, chỉ dùng dữ liệu 5 giờ đầu sau khi bùng phát.

*[Read in English →](README.md)*

---

## Bài toán

Khi một đám cháy rừng bùng phát, chỉ huy hiện trường phải quyết định cảnh báo cộng đồng nào,
cảnh báo lúc nào, và điều lực lượng đi đâu — trong khi chưa có gì chắc chắn. Phần lớn các mô
hình cháy rừng chỉ trả lời một câu hỏi nhị phân: *đám cháy này có nguy hiểm không?* Ứng cứu
khẩn cấp cần nhiều hơn thế. Cần biết **bao lâu nữa**, **tin được đến mức nào**, và **đám nào
xử lý trước**.

Cuộc thi này, tổ chức cùng [Watch Duty](https://www.watchduty.org/), đưa nhu cầu vận hành đó
về bài toán survival analysis. Từ các đặc trưng tính riêng trong **5 giờ đầu** sau lần quan
sát chu vi đầu tiên (`t0`), dự báo xác suất đám cháy tiến vào **trong vòng 5 km quanh tâm một
vùng sơ tán** tại các mốc 12h, 24h, 48h và 72h tính từ `t0 + 5h`.

Nhãn bị kiểm duyệt phải (right-censored):

| | |
|---|---|
| `event = 1` | Cháy đã chạm ngưỡng trong cửa sổ 72h; `time_to_hit_hours` là thời điểm chạm |
| `event = 0` | Bị kiểm duyệt; `time_to_hit_hours` là lần quan sát cuối trong cửa sổ (≤ 72h) |

### Chỉ số đánh giá

```
Hybrid = 0.3 · C-index + 0.7 · (1 − WeightedBrier)
WeightedBrier = 0.3·Brier@24h + 0.4·Brier@48h + 0.3·Brier@72h
```

Brier được tính **có xét kiểm duyệt**: đám cháy bị kiểm duyệt *trước* một mốc thời gian sẽ bị
loại khỏi mốc đó, vì kết cục của nó thực sự không xác định được. Tỉ lệ 70/30 nghĩa là **hiệu
chuẩn quan trọng hơn xếp hạng** — xác suất đủ tin cậy để đặt ngưỡng ra quyết định có giá trị
hơn một thứ tự ưu tiên đúng. Riêng điều này đã định hình gần như toàn bộ thiết kế bên dưới.

---

## Kết quả

| Chỉ số | Out-of-fold |
|---|---|
| **Hybrid score** | **0.97476** |
| C-index | 0.9456 |
| Weighted Brier | 0.01274 |
| Brier @ 12h | 0.05060 |
| Brier @ 24h | 0.02674 |
| Brier @ 48h | 0.01180 |
| Brier @ 72h | 0.00000 |

OOF ≈ 0.9748 tương ứng khoảng 0.970 trên public leaderboard — khoảng cách này giữ ổn định qua
nhiều lần nộp, nhờ vậy tụi mình tin được validation nội bộ để ra quyết định thay vì đốt lượt
submit.

---

## Hướng tiếp cận

Hai họ mô hình có kiểu sai khác nhau, trộn theo từng mốc thời gian.

```
đặc trưng gốc (34) ──► GBSA ensemble          ┐
                       10 configs × 40 seeds   │
                       × 5 folds = 2.000 fit   │
                                               ├─► trộn theo từng mốc
đặc trưng chế biến (54) ► LightGBM + IPCW      │   + hiệu chuẩn lũy thừa cho 24h
                       một mô hình mỗi mốc     │   + đặt 72h = hằng số
                       × 25 seeds × 5 folds   ┘   + sửa tính đơn điệu
                                                         │
                                                         ▼
                                             p12 ≤ p24 ≤ p48 ≤ p72
```

### A. Gradient Boosting Survival Analysis — mô hình xương sống

Xử lý kiểm duyệt phải một cách tự nhiên, điều này rất quan trọng khi **152 trong 221 đám cháy
huấn luyện không bao giờ chạm ngưỡng**. Một lần fit cho ra toàn bộ đường cong sống sót, nên cả
bốn mốc thời gian đều nhất quán với nhau, thay vì phải ghép bốn mô hình rời rạc. Mười cấu hình
được chọn cho *khác nhau có chủ đích* (depth 2–4, learning rate thấp) — sự đa dạng *giữa các
cấu hình* giúp ensemble ổn định hơn nhiều so với việc tinh chỉnh thật kỹ một cấu hình duy nhất.

### B. LightGBM + IPCW — phần sửa hiệu chuẩn

Mỗi mốc thời gian (12h / 24h / 48h) một bộ phân loại nhị phân, chỉ huấn luyện trên các dòng có
kết cục xác định tại mốc đó, kèm trọng số IPCW để khử thiên lệch chọn mẫu. Nếu chỉ đơn giản bỏ
các dòng bị kiểm duyệt thì mô hình sẽ lệch — những đám cháy rời khỏi quan sát sớm không phải
một mẫu ngẫu nhiên — nên mỗi dòng giữ lại được nhân trọng số `1/G(t)`, với `G` là ước lượng
Kaplan–Meier của hàm sống sót *của quá trình kiểm duyệt*.

Họ mô hình này tối ưu đúng thứ mà Brier score đo, tại đúng mốc thời gian được đo. Nó chiếm phần
lớn trọng số trộn ở mốc 48h (0.55) — mốc có trọng số nặng nhất trong chỉ số.

### Trọng số trộn

| Mốc | GBSA | LightGBM | Ghi chú |
|---|---|---|---|
| 12h | 0.97 | 0.03 | Mô hình survival gần như không đối thủ; nhãn thưa nhất |
| 24h | 0.95 | 0.05 | Kèm hiệu chuẩn lũy thừa `p → p^1.1` |
| 48h | 0.45 | 0.55 | Bộ phân loại thắng ở đúng chỗ chỉ số nặng nhất |
| 72h | — | — | Hằng số 1.0 — xem bên dưới |

### Hai quyết định đáng giải thích

**Bất đối xứng đặc trưng.** GBSA nhận các cột *gốc*; LightGBM nhận bảng *đã chế biến*. Trong
ablation, các đặc trưng chế biến làm GBSA **tệ đi** rõ rệt — cây survival tự tìm ra các tương
tác đó, còn các cột cộng tuyến dư thừa chỉ làm tăng phương sai — trong khi chúng lại giúp ích
rõ ràng cho các bộ phân loại nông theo từng mốc.

**Hằng số 72h.** Theo luật tính Brier có xét kiểm duyệt, Brier@72h chỉ chấm hai loại dòng: cháy
đã chạm ngưỡng trước 72h (nhãn 1) và cháy bị kiểm duyệt *sau* 72h (nhãn 0). Trong bộ dữ liệu
này nhóm thứ hai rỗng — kiểm duyệt xảy ra *tại* mốc 72h chứ không phải sau đó — nên mọi dòng
còn được chấm đều là nhãn 1. Dự báo 1.0 cho Brier@72h = 0.0 chính xác, đáng giá ≈ 0.021 điểm
hybrid, và không thể làm hỏng C-index vì một hằng số dịch chuyển điểm rủi ro của mọi đám cháy
như nhau. **Đây là khai thác một đặc thù trong cơ chế kiểm duyệt của bộ dữ liệu này, không phải
vật lý cháy rừng.** Một hệ thống vận hành thật sẽ phải xuất ra xác suất 72h thực. Tụi mình nêu
rõ điều này thay vì tô vẽ nó thành một insight.

### Dòng code giá trị nhất

```python
if gain > 0.001:
    print("nhiều khả năng đang fit nhiễu fold → KHÔNG áp dụng")
```

Với 221 dòng dữ liệu, một mức cải thiện lớn khi tinh chỉnh ba tham số trộn là dấu hiệu cảnh báo
chứ không phải thành công. Tụi mình chỉ nhận những cải thiện OOF nhỏ và ổn định. Quy tắc này
vài lần khiến tụi mình mất vài phần nghìn điểm trên giấy, nhưng đã cứu ít nhất hai lần khỏi
những phương án dễ tụt hạng ở private leaderboard.

---

## Cấu trúc repo

```
.
├── notebooks/
│   └── wids2026_wildfire_survival.ipynb   Lời giải đầy đủ + EDA (chạy thẳng trên Kaggle)
├── src/
│   ├── config.py                          Toàn bộ hằng số cấu hình gom về một chỗ
│   ├── features.py                        Feature engineering
│   ├── metrics.py                         C-index, Brier có xét kiểm duyệt, IPCW, đơn điệu
│   ├── models.py                          Bộ huấn luyện GBSA ensemble + LightGBM IPCW
│   └── pipeline.py                        Nạp dữ liệu → huấn luyện → trộn → nộp bài
├── tests/
│   └── test_smoke.py                      Test trên dữ liệu tổng hợp; không cần file thi
├── docs/
│   └── COMPETITION.md                     Tóm tắt đề bài, chỉ số và mốc thời gian
├── data/                                  Đặt file CSV của cuộc thi vào đây (git-ignored)
├── submissions/                           File nộp bài sinh ra (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

Notebook được viết khép kín để upload thẳng lên Kaggle là chạy được. Thư mục `src/` là chính
logic đó nhưng tách thành module, dành cho ai muốn mở rộng tiếp.

---

## Bắt đầu

```bash
git clone https://github.com/<your-username>/wids-2026-wildfire-survival.git
cd wids-2026-wildfire-survival

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tải dữ liệu từ [trang cuộc thi trên Kaggle](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
và đặt `train.csv`, `test.csv`, `sample_submission.csv` vào thư mục `data/`.

```bash
# Kiểm tra cài đặt — dùng dữ liệu tổng hợp, không cần file thi (~1 phút)
python -m pytest tests/ -v

# Chạy thử nhanh: 10 seeds, ~5 phút
python -m src.pipeline --mode fast

# Chạy nộp bài: 40 seeds GBSA / 25 seeds LightGBM, ~40 phút trên CPU Kaggle
python -m src.pipeline --mode full
```

File nộp bài được ghi vào `submissions/submission.csv` (hoặc `/kaggle/working/` khi chạy trên
Kaggle) và được kiểm tra theo đúng mọi luật của trình validate — schema, khớp ID chính xác,
miền giá trị `[0, 1]`, và tính đơn điệu theo dòng — trước khi ghi ra đĩa.

### Thời gian chạy

| Chế độ | Seeds (GBSA / LGBM) | Số lần fit | Thời gian ước tính |
|---|---|---|---|
| `fast` | 10 / 10 | 500 + 300 | ~5 phút |
| `full` | 40 / 25 | 2.000 + 750 | ~40 phút |

Chỉ cần CPU, không cần GPU. Bộ nhớ đỉnh dưới 2 GB.

---

## Nếu làm tiếp

- **Khoảng tin cậy bằng conformal prediction** cho từng mốc. Người ra quyết định theo ngưỡng
  cần biết khi nào mô hình không chắc chắn, chứ không chỉ biết con số dự báo.
- **Một mô hình 72h thực thụ**, để pipeline không phụ thuộc vào đặc thù kiểm duyệt của bộ dữ
  liệu này.
- **Cross-validation theo vùng địa lý**, để kiểm tra mô hình không đang dựa vào các mẫu hình
  đặc thù của từng khu vực — thứ sẽ không chuyển giao được sang một mùa cháy mới.
- **Ngưỡng có xét chi phí** — bỏ sót một lệnh sơ tán và báo động giả là hai loại sai lầm có chi
  phí hoàn toàn khác nhau, trong khi chỉ số hiện tại đối xử với chúng như nhau.

---

## Nhóm thực hiện

Bốn thành viên, cùng làm trên toàn bộ pipeline thay vì chia phần riêng.

| Tên | Liên kết |
|---|---|
| Trần Ngọc Các Uyên | [GitHub](https://github.com/trnngcccuyn) |
| Nguyễn Hồng Linh | [GitHub](https://github.com/HLiuga05) |
| Nguyễn Diệu Lê | [GitHub](https://github.com/dieule-0810) |
| Lê Minh Phúc Tiên | [GitHub](https://github.com/TienLe-0207) |

---

## Lời cảm ơn

- **[Watch Duty](https://www.watchduty.org/)** — tổ chức phi lợi nhuận đứng sau dữ liệu và cách
  đặt vấn đề của cuộc thi. Họ cung cấp cảnh báo cháy rừng thời gian thực cho hàng triệu người
  trên khắp nước Mỹ, vận hành bởi một đội ngũ nhân sự nhỏ cùng hàng trăm tình nguyện viên là
  lính cứu hỏa, điều phối viên và người trực bộ đàm.
- **[Women in Data Science (WiDS)](https://www.widsconference.org/)** đã tổ chức Global
  Datathon, và **Kaggle** đã đăng cai cuộc thi.
- Xây dựng trên [scikit-survival](https://scikit-survival.readthedocs.io/) và
  [LightGBM](https://lightgbm.readthedocs.io/).

## Giấy phép

Mã nguồn phát hành theo [giấy phép MIT](LICENSE). Bộ dữ liệu của cuộc thi **không** được đính
kèm trong repo này và vẫn tuân theo quy định của WiDS Datathon / Kaggle.
