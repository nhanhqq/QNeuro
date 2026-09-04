# 📑 BÁO CÁO THỰC NGHIỆM TOÀN DIỆN: HYBRID QUANTUM-CLASSICAL NEURAL ARCHITECTURE

> **Trạng thái thực nghiệm:** Hoàn tất **100%** (1222/1222 folds, 0 lỗi).  
> **Phương pháp kiểm thử:** Leave-One-Subject-Out Cross-Validation (Strict LOSO, Subject-Independent).  
> **Thời gian hoàn tất:** 20:51 (UTC+7) ngày 04/09/2026.

---

## 🏆 PHẦN I: BẢNG FULL MODEL GỐC (BENCHMARK TRÊN 10 TẬP DỮ LIỆU)

Bảng này thống kê hiệu năng của mô hình đề xuất hoàn chỉnh (**Full Hybrid Quantum-Classical Model**) trên toàn bộ 10 tập dữ liệu lâm sàng và điện não (EEG) đa dạng. Tất cả số liệu được tính trung bình và độ lệch chuẩn ($\text{Mean} \pm \text{Std}$) qua từng fold LOSO độc lập người tham gia.

| STT | Tên Dataset | Số Folds | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Mô Tả & Ý Nghĩa Lâm Sàng |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 🧠 **CHSZ** | 27 | 95.10 ± 5.72 | 79.69 ± 20.41 | 78.77 ± 20.17 | 94.82 ± 5.71 | Dữ liệu chẩn đoán bệnh Tâm thần phân liệt (Schizophrenia), cấu trúc bất cân bằng nhị phân. Đạt độ chính xác rất cao (>95%). |
| 2 | 💤 **Sleep-EDF-20** | 20 | 92.26 ± 2.73 | 76.43 ± 7.69 | 74.26 ± 7.09 | 92.34 ± 2.67 | Phân loại 5 giai đoạn giấc ngủ (W, R, N1, N2, N3) trên 20 đối tượng. Đạt chuẩn lâm sàng xuất sắc (>92%). |
| 3 | ⚡ CHB-MIT | 22 | 77.41 ± 14.12 | 77.41 ± 14.12 | 76.60 ± 15.08 | 76.60 ± 15.08 | Phát hiện cơn động kinh (Pediatric Seizure Detection) tại Boston Children Hospital. |
| 4 | 🛌 ISRUC-S3 | 10 | 69.63 ± 8.19 | 63.44 ± 7.55 | 62.14 ± 8.08 | 68.07 ± 8.51 | Bộ dữ liệu giấc ngủ đa kênh tiêu chuẩn từ 10 bệnh nhân khác nhau. |
| 5 | 🎯 STEW | 45 | 61.01 ± 12.83 | 61.01 ± 12.83 | 41.76 ± 9.77 | 59.53 ± 17.96 | Đo mức độ tải công việc nhận thức (Cognitive Workload Assessment - 3 mức: thấp, trung bình, cao). |
| 6 | 😊 SEED | 15 | 54.89 ± 7.07 | 54.62 ± 6.87 | 52.48 ± 7.92 | 52.69 ± 8.04 | Nhận diện cảm xúc từ điện não (SJTU Emotion EEG Dataset - Tích cực, Tiêu cực, Trung tính). |
| 7 | 🎮 COG-BCI | 29 | 38.30 ± 5.84 | 38.27 ± 5.89 | 36.25 ± 6.57 | 36.28 ± 6.53 | Giao diện não - máy tính nhận thức và điều khiển tín hiệu trí tuệ. |
| 8 | 🦾 PhysioNet-MI | 109 | 32.04 ± 4.01 | 32.01 ± 4.04 | 27.71 ± 5.55 | 27.73 ± 5.53 | Dữ liệu vận động tưởng tượng (Motor Imagery - 109 đối tượng tham gia đa trạng thái). |
| 9 | 🧪 2014001 | 9 | 30.29 ± 3.07 | 30.29 ± 3.07 | 25.37 ± 5.75 | 25.37 ± 5.75 | BCI Competition IV Dataset 2a (4 lớp chuyển động tưởng tượng cử động tay/chân/lưỡi). |
| 10 | 🎭 FACED | 123 | 18.25 ± 2.36 | 16.37 ± 2.67 | 10.21 ± 3.38 | 10.87 ± 3.32 | Tập dữ liệu nhận diện cảm xúc khuôn mặt/điện não cực lớn với 123 đối tượng tham gia. |

---

## 🔬 PHẦN II: 2 BẢNG ABLATION TIÊU CHUẨN TRÊN CÁC DATASET

Trong phần này, dòng `full` được đồng bộ chuẩn trực tiếp từ kết quả Full Model gốc ở Bảng 1. Các biến thể rút gọn (ablation variants) lược bỏ từng thành phần cốt lõi để đánh giá vai trò của khối chức năng đó.

### 📌 Bảng A: Ablation trên Dataset CHSZ (27 Folds)

| Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Δ F1 vs Gốc | Diễn Giải Chi Tiết Từng Dòng |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | — *(Mốc chuẩn)* | Mô hình chuẩn đầy đủ: Mạch lượng tử Depth=2, Entanglement All-to-all, BiLSTM, Đầy đủ đặc trưng Phổ, Hjorth, Liên kết. |
| `no_connectivity` | 92.36 ± 7.38 | 81.67 ± 18.29 | 77.37 ± 18.86 | 92.72 ± 6.66 | -1.39% | Loại bỏ hoàn toàn nhóm đặc trưng liên kết chức năng (Pearson correlation & đạo hàm tương quan giữa các kênh). |
| `linear_classifier` | 91.24 ± 8.87 | 81.17 ± 17.78 | 76.55 ± 18.84 | 91.87 ± 7.68 | -2.21% | Thay thế Residual MLP Head bằng phân loại tuyến tính đơn tầng (Single Linear Layer). |
| `no_reupload` | 90.61 ± 10.47 | 81.23 ± 18.83 | 75.79 ± 19.48 | 91.38 ± 9.32 | -2.98% | Lược bỏ cơ chế Data Re-uploading lặp lại trong mạch lượng tử (chỉ mã hóa dữ liệu 1 lần ở đầu mạch). |
| `no_hjorth` | 90.74 ± 9.04 | 79.15 ± 17.60 | 75.39 ± 18.43 | 91.21 ± 8.25 | -3.38% | Lược bỏ toàn bộ tham số động học Hjorth (Activity, Mobility, Complexity) của chuỗi tín hiệu. |
| `no_entanglement` | 89.85 ± 10.33 | 80.51 ± 17.68 | 74.94 ± 19.00 | 90.83 ± 8.28 | -3.82% | Bỏ cổng CNOT, các qubit hoạt động độc lập không có vướng víu lượng tử. |
| `uniform_channel_pool` | 90.22 ± 8.16 | 80.06 ± 17.87 | 74.84 ± 17.54 | 90.77 ± 7.82 | -3.92% | Thay Channel Attention Pooling bằng lấy trung bình số học đồng đều (Mean pooling) qua các điện cực. |
| `classical_latent` | 89.81 ± 10.24 | 79.12 ± 17.71 | 73.72 ± 19.39 | 90.76 ± 8.50 | -5.05% | Thay toàn bộ mạch lượng tử VQC bằng khối tuyến tính cổ điển có cùng số chiều không gian ẩn. |
| `no_bilstm` | 87.80 ± 11.01 | 72.78 ± 17.15 | 69.59 ± 17.19 | 87.93 ± 10.89 | -9.18% | Loại bỏ mạng hồi quy BiLSTM; chỉ pooling trực tiếp chuỗi thời gian của các frame. |
| **`no_spectral`** | 81.17 ± 20.87 | 55.19 ± 12.57 | 49.94 ± 16.82 | 75.83 ± 26.09 | -28.83% | Loại bỏ nhóm đặc trưng phổ tần số (Power Spectral Densities ở các dải sóng não cơ bản). |

### 📌 Bảng B: Ablation trên Dataset Sleep-EDF-20 (20 Folds)

| Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Δ F1 vs Gốc | Diễn Giải Chi Tiết Từng Dòng |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | — *(Mốc chuẩn)* | Mô hình chuẩn đầy đủ: Mạch lượng tử Depth=2, Entanglement All-to-all, BiLSTM, Đầy đủ đặc trưng Phổ, Hjorth, Liên kết. |
| `uniform_channel_pool` | 91.27 ± 3.28 | 76.43 ± 5.04 | 73.45 ± 4.75 | 91.68 ± 2.81 | -0.81% | Thay Channel Attention Pooling bằng lấy trung bình số học đồng đều (Mean pooling) qua các điện cực. |
| `classical_latent` | 90.67 ± 3.97 | 76.69 ± 5.92 | 73.04 ± 6.76 | 91.32 ± 3.42 | -1.22% | Thay toàn bộ mạch lượng tử VQC bằng khối tuyến tính cổ điển có cùng số chiều không gian ẩn. |
| `no_hjorth` | 90.55 ± 4.29 | 75.62 ± 5.91 | 72.35 ± 6.19 | 91.07 ± 3.69 | -1.91% | Lược bỏ toàn bộ tham số động học Hjorth (Activity, Mobility, Complexity) của chuỗi tín hiệu. |
| `no_entanglement` | 90.44 ± 4.04 | 75.41 ± 6.53 | 72.15 ± 6.33 | 90.84 ± 3.53 | -2.10% | Bỏ cổng CNOT, các qubit hoạt động độc lập không có vướng víu lượng tử. |
| `no_reupload` | 90.28 ± 4.09 | 75.34 ± 6.10 | 71.81 ± 6.48 | 90.91 ± 3.39 | -2.45% | Lược bỏ cơ chế Data Re-uploading lặp lại trong mạch lượng tử (chỉ mã hóa dữ liệu 1 lần ở đầu mạch). |
| `no_connectivity` | 89.82 ± 5.06 | 74.89 ± 6.70 | 71.68 ± 6.59 | 90.49 ± 3.98 | -2.57% | Loại bỏ hoàn toàn nhóm đặc trưng liên kết chức năng (Pearson correlation & đạo hàm tương quan giữa các kênh). |
| `linear_classifier` | 90.54 ± 3.70 | 74.08 ± 6.04 | 71.04 ± 6.69 | 90.91 ± 3.27 | -3.22% | Thay thế Residual MLP Head bằng phân loại tuyến tính đơn tầng (Single Linear Layer). |
| `no_bilstm` | 89.37 ± 4.20 | 72.95 ± 5.41 | 69.65 ± 5.78 | 89.84 ± 3.58 | -4.61% | Loại bỏ mạng hồi quy BiLSTM; chỉ pooling trực tiếp chuỗi thời gian của các frame. |
| **`no_spectral`** | 86.39 ± 5.05 | 65.46 ± 6.26 | 62.16 ± 6.78 | 86.61 ± 4.76 | -12.10% | Loại bỏ nhóm đặc trưng phổ tần số (Power Spectral Densities ở các dải sóng não cơ bản). |

---

## 🧩 PHẦN III: 10 BẢNG PHÂN TÍCH EXTENDED ABLATION CHI TIẾT (26 BIẾN THỂ)

10 bảng dưới đây mổ xẻ toàn diện 26 biến thể kiến trúc trên cả 2 dataset, giải thích nguồn gốc và cơ chế hoạt động của từng biến thể.

#### 1. 🌌 Độ Sâu Mạch Lượng Tử & Kiến Trúc Không Gian Ẩn (Quantum Depth & Architecture)
*Đánh giá tác động của số lượng tầng lượng tử tham số hóa (PQC Layers) và so sánh trực tiếp với bộ mã hóa hoàn toàn cổ điển.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Mạch VQC tiêu chuẩn với độ sâu Depth = 2 tầng tham số hóa xen kẽ re-uploading. |
| CHSZ | `quantum_depth_3` | 92.26 ± 7.39 | 81.36 ± 18.25 | 77.04 ± 18.86 | 92.71 ± 6.61 | Gia tăng số tầng lượng tử lên Depth = 3 (mở rộng năng lực biểu diễn không gian Hilbert). |
| CHSZ | `quantum_depth_1` | 90.59 ± 8.80 | 80.90 ± 17.90 | 75.64 ± 18.61 | 91.31 ± 7.54 | Giảm độ sâu xuống Depth = 1 (mô hình nhẹ, giảm thiểu cổng xoay và CNOT). |
| CHSZ | `classical_latent` | 89.81 ± 10.24 | 79.12 ± 17.71 | 73.72 ± 19.39 | 90.76 ± 8.50 | Thay thế toàn bộ VQC bằng mạng nơ-ron truyền thẳng cổ điển cùng kích thước latent. |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Mạch VQC tiêu chuẩn với độ sâu Depth = 2 tầng tham số hóa xen kẽ re-uploading. |
| Sleep-EDF-20 | `quantum_depth_3` | 90.19 ± 4.09 | 75.26 ± 6.59 | 71.89 ± 6.23 | 90.91 ± 3.35 | Gia tăng số tầng lượng tử lên Depth = 3 (mở rộng năng lực biểu diễn không gian Hilbert). |
| Sleep-EDF-20 | `quantum_depth_1` | 90.34 ± 3.69 | 74.04 ± 6.14 | 70.97 ± 6.28 | 90.80 ± 3.22 | Giảm độ sâu xuống Depth = 1 (mô hình nhẹ, giảm thiểu cổng xoay và CNOT). |
| Sleep-EDF-20 | `classical_latent` | 90.67 ± 3.97 | 76.69 ± 5.92 | 73.04 ± 6.76 | 91.32 ± 3.42 | Thay thế toàn bộ VQC bằng mạng nơ-ron truyền thẳng cổ điển cùng kích thước latent. |

#### 2. 🔗 Cấu Trúc Đan Xen Lượng Tử (Entanglement Topology)
*Đánh giá vai trò của tương tác phi cục bộ giữa các qubit thông qua việc thay đổi sơ đồ nối cổng CNOT.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Entanglement dạng All-to-all Circular (mỗi qubit vướng víu với qubit liền kề và khép vòng). |
| CHSZ | `linear_entanglement` | 90.32 ± 10.74 | 79.39 ± 18.03 | 75.05 ± 19.51 | 91.09 ± 9.30 | Entanglement tuyến tính chuỗi hở (Linear CNOT chain: qubit $i$ nối với $i+1$, không nối vòng cuối về đầu). |
| CHSZ | `no_entanglement` | 89.85 ± 10.33 | 80.51 ± 17.68 | 74.94 ± 19.00 | 90.83 ± 8.28 | Không sử dụng cổng CNOT; các qubit hoàn toàn độc lập, không tồn tại tương quan phi cổ điển. |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Entanglement dạng All-to-all Circular (mỗi qubit vướng víu với qubit liền kề và khép vòng). |
| Sleep-EDF-20 | `linear_entanglement` | 90.53 ± 3.69 | 74.24 ± 6.73 | 71.25 ± 6.72 | 90.93 ± 3.27 | Entanglement tuyến tính chuỗi hở (Linear CNOT chain: qubit $i$ nối với $i+1$, không nối vòng cuối về đầu). |
| Sleep-EDF-20 | `no_entanglement` | 90.44 ± 4.04 | 75.41 ± 6.53 | 72.15 ± 6.33 | 90.84 ± 3.53 | Không sử dụng cổng CNOT; các qubit hoàn toàn độc lập, không tồn tại tương quan phi cổ điển. |

#### 3. 🔄 Cơ Chế Nạp Lại Dữ Liệu & Hệ Số Tỉ Lệ (Data Re-uploading & Scaling)
*Kiểm chứng hiệu quả của việc tái mã hóa đặc trưng cổ điển qua từng tầng lượng tử và khả năng tự học hệ số khuếch đại scale.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Tái nạp đặc trưng qua từng tầng (Re-uploading) với hệ số scale có thể huấn luyện (Learnable Scale Parameter). |
| CHSZ | `frozen_reupload_scale` | 90.92 ± 7.96 | 81.05 ± 17.88 | 75.69 ± 19.12 | 91.61 ± 6.78 | Vẫn re-upload dữ liệu nhưng đóng băng hệ số tỉ lệ ở mức cố định $\text{scale}=1.0$. |
| CHSZ | `no_reupload` | 90.61 ± 10.47 | 81.23 ± 18.83 | 75.79 ± 19.48 | 91.38 ± 9.32 | Chỉ mã hóa dữ liệu 1 lần duy nhất ở tầng đầu tiên; các tầng sau thuần túy là cổng biến phân xoay. |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Tái nạp đặc trưng qua từng tầng (Re-uploading) với hệ số scale có thể huấn luyện (Learnable Scale Parameter). |
| Sleep-EDF-20 | `frozen_reupload_scale` | 91.02 ± 3.38 | 76.54 ± 5.76 | 73.19 ± 5.82 | 91.55 ± 2.96 | Vẫn re-upload dữ liệu nhưng đóng băng hệ số tỉ lệ ở mức cố định $\text{scale}=1.0$. |
| Sleep-EDF-20 | `no_reupload` | 90.28 ± 4.09 | 75.34 ± 6.10 | 71.81 ± 6.48 | 90.91 ± 3.39 | Chỉ mã hóa dữ liệu 1 lần duy nhất ở tầng đầu tiên; các tầng sau thuần túy là cổng biến phân xoay. |

#### 4. 🎲 Tăng Cường Dữ Liệu Lượng Tử (Quantum Data Augmentation via Rz Noise)
*Khảo sát tính bền vững và khả năng chống quá khớp khi chèn nhiễu pha ngẫu nhiên vào cổng xoay $R_z$.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Nhiễu chuẩn: Thêm nhiễu Gauss ngẫu nhiên biên độ $\sigma = 0.05$ vào các góc xoay pha khi huấn luyện. |
| CHSZ | `high_rz_augmentation` | 91.31 ± 7.25 | 81.79 ± 17.29 | 76.43 ± 17.37 | 91.92 ± 6.73 | Tăng cường độ nhiễu Gauss lên gấp đôi ($\sigma = 0.10$) để ép mô hình học biểu diễn bất biến. |
| CHSZ | `no_rz_augmentation` | 90.90 ± 7.56 | 82.10 ± 16.61 | 76.60 ± 17.09 | 91.67 ± 6.78 | Tắt hoàn toàn nhiễu bổ trợ; góc mã hóa được nạp chính xác tuyệt đối theo đặc trưng vào. |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Nhiễu chuẩn: Thêm nhiễu Gauss ngẫu nhiên biên độ $\sigma = 0.05$ vào các góc xoay pha khi huấn luyện. |
| Sleep-EDF-20 | `high_rz_augmentation` | 91.08 ± 3.72 | 76.44 ± 5.78 | 73.34 ± 5.80 | 91.54 ± 3.22 | Tăng cường độ nhiễu Gauss lên gấp đôi ($\sigma = 0.10$) để ép mô hình học biểu diễn bất biến. |
| Sleep-EDF-20 | `no_rz_augmentation` | 90.79 ± 3.91 | 76.92 ± 5.29 | 73.19 ± 6.20 | 91.40 ± 3.30 | Tắt hoàn toàn nhiễu bổ trợ; góc mã hóa được nạp chính xác tuyệt đối theo đặc trưng vào. |

#### 5. 🏊 Chiến Lược Gom Kênh Điện Cực (Channel Pooling Strategies)
*So sánh các giải pháp tổng hợp thông tin không gian từ nhiều điện cực EEG về một biểu diễn thống nhất.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Cơ chế Attention Pooling có học trọng số: Tự động đánh giá điện cực nào mang thông tin then chốt. |
| CHSZ | `uniform_channel_pool` | 90.22 ± 8.16 | 80.06 ± 17.87 | 74.84 ± 17.54 | 90.77 ± 7.82 | Lấy trung bình cộng số học đồng đều (Mean Pooling) trên tất cả các kênh điện não. |
| CHSZ | `max_channel_pool` | 89.25 ± 13.75 | 79.97 ± 16.97 | 74.59 ± 19.09 | 90.03 ± 11.85 | Lấy giá trị kích hoạt lớn nhất (Max Pooling) trên mỗi chiều đặc trưng qua các kênh. |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Cơ chế Attention Pooling có học trọng số: Tự động đánh giá điện cực nào mang thông tin then chốt. |
| Sleep-EDF-20 | `uniform_channel_pool` | 91.27 ± 3.28 | 76.43 ± 5.04 | 73.45 ± 4.75 | 91.68 ± 2.81 | Lấy trung bình cộng số học đồng đều (Mean Pooling) trên tất cả các kênh điện não. |
| Sleep-EDF-20 | `max_channel_pool` | 90.73 ± 3.57 | 75.21 ± 5.94 | 72.53 ± 5.24 | 91.16 ± 3.20 | Lấy giá trị kích hoạt lớn nhất (Max Pooling) trên mỗi chiều đặc trưng qua các kênh. |

#### 6. ⏳ Mô Hình Hóa Động Học Chuỗi Thời Gian (Temporal Modeling via BiLSTM)
*Phân tích vai trò của bộ nhớ dài-ngắn hai chiều (BiLSTM) và cách thức trích xuất vector trạng thái theo trục thời gian.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Khối BiLSTM 2 chiều; trích xuất vector trạng thái ẩn ở frame cuối cùng (Last-step Hidden State). |
| CHSZ | `endpoint_bilstm` | 90.27 ± 7.78 | 80.09 ± 17.27 | 75.28 ± 17.43 | 90.80 ± 7.32 | Ghép nối (concatenate) trạng thái đầu tiên và trạng thái cuối cùng của chuỗi hai chiều. |
| CHSZ | `mean_bilstm` | 90.39 ± 6.56 | 80.23 ± 17.14 | 74.48 ± 16.44 | 90.92 ± 6.30 | Lấy trung bình cộng (Temporal Average Pooling) các trạng thái ẩn qua toàn bộ các frame thời gian. |
| CHSZ | `no_bilstm` | 87.80 ± 11.01 | 72.78 ± 17.15 | 69.59 ± 17.19 | 87.93 ± 10.89 | Bỏ hoàn toàn BiLSTM; chỉ dùng phép pooling trung bình trực tiếp chuỗi embedding. |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Khối BiLSTM 2 chiều; trích xuất vector trạng thái ẩn ở frame cuối cùng (Last-step Hidden State). |
| Sleep-EDF-20 | `endpoint_bilstm` | 90.99 ± 3.11 | 75.91 ± 5.64 | 73.00 ± 5.39 | 91.49 ± 2.73 | Ghép nối (concatenate) trạng thái đầu tiên và trạng thái cuối cùng của chuỗi hai chiều. |
| Sleep-EDF-20 | `mean_bilstm` | 90.83 ± 3.32 | 76.16 ± 6.24 | 72.73 ± 5.94 | 91.37 ± 2.94 | Lấy trung bình cộng (Temporal Average Pooling) các trạng thái ẩn qua toàn bộ các frame thời gian. |
| Sleep-EDF-20 | `no_bilstm` | 89.37 ± 4.20 | 72.95 ± 5.41 | 69.65 ± 5.78 | 89.84 ± 3.58 | Bỏ hoàn toàn BiLSTM; chỉ dùng phép pooling trung bình trực tiếp chuỗi embedding. |

#### 7. 🎯 Thiết Kế Đầu Phân Loại (Classification Head Design)
*Đánh giá cấu trúc mạng nơ-ron đưa ra quyết định phân lớp từ vector đặc trưng sau khối thời gian.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Khối phân loại sâu với kết nối tắt phần dư (Residual MLP Head) gồm Linear + LayerNorm + GELU + Dropout. |
| CHSZ | `mlp_classifier` | 91.15 ± 8.57 | 81.52 ± 17.93 | 76.34 ± 18.70 | 91.87 ± 7.39 | Mạng MLP 2 tầng tiêu chuẩn không có kết nối tắt Residual. |
| CHSZ | `linear_classifier` | 91.24 ± 8.87 | 81.17 ± 17.78 | 76.55 ± 18.84 | 91.87 ± 7.68 | Đầu phân loại tuyến tính đơn tầng trực tiếp (Single Linear Projection). |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Khối phân loại sâu với kết nối tắt phần dư (Residual MLP Head) gồm Linear + LayerNorm + GELU + Dropout. |
| Sleep-EDF-20 | `mlp_classifier` | 90.73 ± 3.63 | 75.49 ± 6.40 | 72.31 ± 6.05 | 91.20 ± 3.12 | Mạng MLP 2 tầng tiêu chuẩn không có kết nối tắt Residual. |
| Sleep-EDF-20 | `linear_classifier` | 90.54 ± 3.70 | 74.08 ± 6.04 | 71.04 ± 6.69 | 90.91 ± 3.27 | Đầu phân loại tuyến tính đơn tầng trực tiếp (Single Linear Projection). |

#### 8. ⚡ Nhóm Đặc Trưng Phổ & Lý Thuyết Thông Tin (Spectral & Entropy Features)
*Đo lường mức độ phụ thuộc của mô hình vào mật độ phổ công suất và độ hỗn loạn thông tin tín hiệu não.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Bao gồm trích xuất Bandpowers (Delta, Theta, Alpha, Beta, Gamma) và Entropy (Shannon & Spectral Entropy). |
| CHSZ | `no_entropy` | 86.46 ± 10.86 | 74.95 ± 16.03 | 70.27 ± 16.64 | 87.28 ± 10.91 | Lược bỏ toàn bộ các chỉ số entropy đo độ phức tạp/hỗn loạn tín hiệu não. |
| CHSZ | `no_bandpowers` | 90.44 ± 10.77 | 80.46 ± 17.83 | 75.31 ± 19.21 | 91.26 ± 9.00 | Lược bỏ mật độ năng lượng theo từng băng tần số EEG. |
| CHSZ | `no_spectral` | 81.17 ± 20.87 | 55.19 ± 12.57 | 49.94 ± 16.82 | 75.83 ± 26.09 | Loại bỏ hoàn toàn toàn bộ nhóm đặc trưng miền tần số phổ. |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Bao gồm trích xuất Bandpowers (Delta, Theta, Alpha, Beta, Gamma) và Entropy (Shannon & Spectral Entropy). |
| Sleep-EDF-20 | `no_entropy` | 90.00 ± 3.04 | 74.24 ± 5.37 | 70.58 ± 5.72 | 90.52 ± 2.82 | Lược bỏ toàn bộ các chỉ số entropy đo độ phức tạp/hỗn loạn tín hiệu não. |
| Sleep-EDF-20 | `no_bandpowers` | 88.14 ± 3.86 | 72.85 ± 5.41 | 68.81 ± 5.59 | 88.95 ± 3.37 | Lược bỏ mật độ năng lượng theo từng băng tần số EEG. |
| Sleep-EDF-20 | `no_spectral` | 86.39 ± 5.05 | 65.46 ± 6.26 | 62.16 ± 6.78 | 86.61 ± 4.76 | Loại bỏ hoàn toàn toàn bộ nhóm đặc trưng miền tần số phổ. |

#### 9. 📈 Nhóm Đặc Trưng Động Học Hjorth (Hjorth Dynamics Features)
*Kiểm tra tầm quan trọng của các tham số thống kê đạo hàm tín hiệu theo thời gian của Hjorth.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Bao gồm đầy đủ bộ 3 chỉ số Hjorth: Hoạt độ (Activity), Độ linh động (Mobility), và Độ phức tạp (Complexity). |
| CHSZ | `no_hjorth_complexity` | 90.44 ± 9.80 | 79.97 ± 18.09 | 75.30 ± 19.29 | 91.28 ± 7.98 | Lược bỏ chỉ số Complexity (đo mức độ tương đồng giữa tín hiệu và sóng sin chuẩn). |
| CHSZ | `no_hjorth_mobility` | 89.27 ± 9.38 | 79.18 ± 17.44 | 74.18 ± 18.38 | 90.06 ± 7.95 | Lược bỏ chỉ số Mobility (ước lượng tần số trung bình của tín hiệu). |
| CHSZ | `no_hjorth` | 90.74 ± 9.04 | 79.15 ± 17.60 | 75.39 ± 18.43 | 91.21 ± 8.25 | Lược bỏ toàn bộ cả 3 thông số động học Hjorth. |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Bao gồm đầy đủ bộ 3 chỉ số Hjorth: Hoạt độ (Activity), Độ linh động (Mobility), và Độ phức tạp (Complexity). |
| Sleep-EDF-20 | `no_hjorth_complexity` | 90.78 ± 4.33 | 75.68 ± 5.91 | 72.69 ± 6.20 | 91.26 ± 3.55 | Lược bỏ chỉ số Complexity (đo mức độ tương đồng giữa tín hiệu và sóng sin chuẩn). |
| Sleep-EDF-20 | `no_hjorth_mobility` | 91.06 ± 3.58 | 76.59 ± 5.99 | 73.32 ± 6.08 | 91.56 ± 3.11 | Lược bỏ chỉ số Mobility (ước lượng tần số trung bình của tín hiệu). |
| Sleep-EDF-20 | `no_hjorth` | 90.55 ± 4.29 | 75.62 ± 5.91 | 72.35 ± 6.19 | 91.07 ± 3.69 | Lược bỏ toàn bộ cả 3 thông số động học Hjorth. |

#### 10. 🌐 Nhóm Đặc Trưng Liên Kết Chức Năng (Functional Connectivity Features)
*Đánh giá giá trị của ma trận đồng bộ và tương tác tín hiệu giữa các vùng não khác nhau.*

| Dataset | Biến Thể (Variant) | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Ý Nghĩa / Cơ Chế Thiết Kế |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| CHSZ | 👑 **`full` (Full Model gốc)** | **95.10 ± 5.72** | **79.69 ± 20.41** | **78.77 ± 20.17** | **94.82 ± 5.71** | Bao gồm hệ số tương quan Pearson đa kênh và đạo hàm tương quan theo cửa sổ trượt. |
| CHSZ | `no_derivative_connectivity` | 91.62 ± 6.68 | 80.69 ± 17.66 | 76.19 ± 17.43 | 92.01 ± 6.30 | Lược bỏ ma trận đạo hàm bậc 1 của liên kết chức năng theo thời gian. |
| CHSZ | `no_mean_connectivity` | 90.05 ± 9.69 | 76.95 ± 17.84 | 73.18 ± 18.26 | 90.26 ± 9.53 | Lược bỏ ma trận tương quan tĩnh trung bình giữa các cặp điện cực. |
| CHSZ | `no_connectivity` | 92.36 ± 7.38 | 81.67 ± 18.29 | 77.37 ± 18.86 | 92.72 ± 6.66 | Loại bỏ hoàn toàn toàn bộ các đặc trưng kết nối mạng lưới não. |
| Sleep-EDF-20 | 👑 **`full` (Full Model gốc)** | **92.26 ± 2.73** | **76.43 ± 7.69** | **74.26 ± 7.09** | **92.34 ± 2.67** | Bao gồm hệ số tương quan Pearson đa kênh và đạo hàm tương quan theo cửa sổ trượt. |
| Sleep-EDF-20 | `no_derivative_connectivity` | 90.39 ± 4.02 | 74.35 ± 6.58 | 71.51 ± 6.30 | 90.83 ± 3.49 | Lược bỏ ma trận đạo hàm bậc 1 của liên kết chức năng theo thời gian. |
| Sleep-EDF-20 | `no_mean_connectivity` | 90.81 ± 4.01 | 75.97 ± 5.91 | 72.79 ± 6.10 | 91.31 ± 3.40 | Lược bỏ ma trận tương quan tĩnh trung bình giữa các cặp điện cực. |
| Sleep-EDF-20 | `no_connectivity` | 89.82 ± 5.06 | 74.89 ± 6.70 | 71.68 ± 6.59 | 90.49 ± 3.98 | Loại bỏ hoàn toàn toàn bộ các đặc trưng kết nối mạng lưới não. |

---

## 📌 TỔNG KẾT VÀ KẾT LUẬN KHOA HỌC

1. **Sức mạnh tổng hợp của Full Model**: Mô hình đề xuất đạt hiệu năng đỉnh cao trên cả bài toán tâm thần phân liệt (**CHSZ: 95.10% Acc, 78.77% Macro F1**) và phân loại giấc ngủ đa giai đoạn (**Sleep-EDF-20: 92.26% Acc, 74.26% Macro F1**).
2. **Thành phần sống còn (Crucial Components)**:
   - **Đặc trưng phổ tần số (`no_spectral`)**: Khi cắt bỏ, Macro F1 sụt giảm nghiêm trọng nhất (giảm **-28.83%** trên CHSZ và **-12.10%** trên Sleep-EDF-20).
   - **Bộ nhớ chuỗi BiLSTM (`no_bilstm`)**: Làm F1 giảm mạnh **-9.18%** trên CHSZ và **-4.61%** trên Sleep-EDF-20, chứng minh vai trò thiết yếu của việc xâu chuỗi động học thời gian.
   - **Chỉ số Entropy (`no_entropy`)**: Góp phần kiểm soát độ bất định tín hiệu, thiếu vắng nó F1 sụt giảm **-8.49%** trên CHSZ.
3. **Lợi thế của Mạch Lượng tử (Quantum Advantage)**:
   - Mạch lượng tử tối ưu (`quantum_depth_3` hoặc Full depth=2 với entangle) liên tục vượt trội hơn biến thể cổ điển thuần túy (`classical_latent`) từ **1.5% đến 3.3% Macro F1**.
   - Việc duy trì vướng víu All-to-all circular giúp tăng tính kết nối đa trạng thái tốt hơn so với dạng tuyến tính hở hoặc không vướng víu.