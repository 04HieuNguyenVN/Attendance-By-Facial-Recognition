# Prompts để Viết Báo cáo Môn Trí tuệ Nhân tạo
Dưới đây là bộ các prompt chi tiết, được chia thành từng phần theo khung báo cáo bạn đã cung cấp. Các prompt này được xây dựng để tạo ra nội dung bám sát vào đề cương của bạn.

Bạn có thể sao chép từng prompt và đưa cho một mô hình ngôn ngữ lớn (như tôi) để tạo nội dung cho báo cáo.

---

## CHƯƠG 1: CƠ SỞ LÝ THUYẾT

### **Prompt 1.1: Tổng quan về trí tuệ nhân tạo và nhận diện khuôn mặt**

**Yêu cầu:**
"Hãy viết phần **1.1: Tổng quan về trí tuệ nhân tạo và nhận diện khuôn mặt** cho báo cáo môn học Trí tuệ nhân tạo. Nội dung cần được trình bày một cách học thuật, rõ ràng, bằng tiếng Việt.

Dựa trên các điểm sau:
1.  **Khái niệm cơ bản:** Định nghĩa Trí tuệ nhân tạo (AI), Học máy (Machine Learning), và Học sâu (Deep Learning).
2.  **Mối quan hệ phân cấp:** Giải thích mối quan hệ giữa các khái niệm này (AI > ML > Deep Learning).
3.  **Bài toán nhận diện khuôn mặt:** Giới thiệu bài toán nhận diện khuôn mặt như một ứng dụng tiêu biểu của AI và Computer Vision, nêu các thách thức thực tế (góc mặt, ánh sáng, biểu cảm).
4.  **Ứng dụng của AI trong giáo dục:** Thảo luận ngắn gọn về vai trò của AI trong việc tự động hóa các quy trình quản lý giáo dục như điểm danh."

---

### **Prompt 1.2: Tổng quan về xử lý ảnh và thị giác máy tính**

**Yêu cầu:**
"Hãy viết phần **1.2: Tổng quan về xử lý ảnh và thị giác máy tính** cho báo cáo.

Nội dung cần tập trung vào các khía cạnh sau:
1.  **Khái niệm:** Giới thiệu ngắn gọn về Xử lý ảnh và Thị giác máy tính, phân biệt sự khác nhau cơ bản giữa chúng.
2.  **Vị trí của Nhận diện khuôn mặt:** Mô tả bài toán nhận diện khuôn mặt là một nhiệm vụ cốt lõi của Computer Vision.
3.  **Các bước cơ bản trong pipeline nhận diện:** Trình bày các bước chuẩn của một hệ thống nhận diện khuôn mặt: Phát hiện khuôn mặt (Face Detection) → Căn chỉnh khuôn mặt (Face Alignment) → Trích xuất đặc trưng (Feature Extraction) → So khớp (Matching/Verification)."

---

### **Prompt 1.3: Giới thiệu mô hình sử dụng trong đề tài**

**Yêu cầu:**
"Hãy viết phần **1.3: Giới thiệu mô hình sử dụng trong đề tài**. Tập trung vào mô hình ResNet-based CNN từ thư viện Dlib (`face_recognition`).

Nội dung cần bao gồm:
1.  **Nền tảng CNN và ResNet:**
    *   Giới thiệu ngắn gọn về Mạng Nơ-ron Tích chập (CNN) là nền tảng cho các bài toán xử lý ảnh.
    *   Giới thiệu Mạng Nơ-ron Phần dư (ResNet) như một kiến trúc CNN nâng cao, cho phép huấn luyện mạng sâu và chính xác hơn.
2.  **Mô hình chính: Dlib Face Recognition Model:**
    *   Giới thiệu mô hình được cung cấp bởi thư viện `face_recognition` (dựa trên Dlib) là mô hình chính của dự án.
    *   Mô tả mô hình này dựa trên kiến trúc ResNet, tạo ra một **vector embedding 128 chiều**.
    *   Giải thích rằng quá trình so khớp của mô hình này dựa trên **khoảng cách Euclidean**: khoảng cách càng nhỏ thì hai khuôn mặt càng giống nhau, và thường sử dụng một ngưỡng (ví dụ: 0.6) để quyết định có khớp hay không."

---

## CHƯƠNG 2: PHÁT BIỂU GIẢI QUYẾT BÀI TOÁN / VẤN ĐỀ

### **Prompt 2.1 & 2.2: Bài toán thực tế và Mục tiêu giải quyết**

**Yêu cầu:**
"Hãy viết kết hợp phần **2.1: Bài toán / Vấn đề thực tế** và **2.2: Mục tiêu giải quyết** cho báo cáo, bám sát các ý sau:

1.  **Vấn đề thực tế:**
    *   Mô tả những hạn chế của phương pháp điểm danh thủ công: tốn thời gian, gây gián đoạn, dễ sai sót và có thể xảy ra gian lận (điểm danh hộ).
    *   Phân tích ngắn gọn quy trình điểm danh hiện tại ở các trường đại học và chỉ ra các 'điểm đau' (pain points).
    *   Nhấn mạnh nhu cầu cấp thiết về một giải pháp tự động hóa để tăng hiệu quả và độ tin cậy.
2.  **Mục tiêu của đề tài:**
    *   Trình bày mục tiêu chính: Xây dựng một hệ thống có khả năng tự động nhận diện khuôn mặt sinh viên từ luồng video camera thời gian thực.
    *   Liệt kê các mục tiêu cụ thể:
        *   Ghi nhận thông tin điểm danh (tên sinh viên, thời gian có mặt) một cách chính xác và tự động.
        *   Cung cấp chức năng thống kê, cho phép xem lại lịch sử điểm danh và số buổi học sinh viên đã tham gia."

---

### **Prompt 2.3 & 2.4: Phát biểu bài toán AI và Đầu vào - Đầu ra**

**Yêu cầu:**
"Hãy viết phần **2.3: Phát biểu bài toán AI** và **2.4: Đầu vào – Đầu ra** cho báo cáo, theo đúng cấu trúc sau:

1.  **Phát biểu bài toán AI:**
    *   Sử dụng phát biểu sau: *'Cho một ảnh/video chứa khuôn mặt người, hệ thống cần xác định danh tính của sinh viên dựa trên đặc trưng khuôn mặt và tự động ghi nhận sự có mặt của sinh viên trong buổi học.'*
2.  **Mô tả Đầu vào - Xử lý - Đầu ra:**
    *   Trình bày dưới dạng bảng hoặc danh sách rõ ràng:
        *   **Đầu vào (Input):** Luồng video thời gian thực từ camera, hình ảnh khuôn mặt của sinh viên đã đăng ký, và cơ sở dữ liệu chứa các vector embedding (128 chiều) đã được tính toán trước.
        *   **Xử lý (Processing):** Mô tả pipeline xử lý một khung hình theo các bước: Phát hiện khuôn mặt (Face Detection) → Căn chỉnh khuôn mặt (Face Alignment) → Trích xuất vector đặc trưng 128 chiều bằng mô hình CNN (Feature Extraction) → So sánh vector bằng khoảng cách Euclidean (Matching) → Nếu khoảng cách nhỏ hơn ngưỡng, ghi nhận điểm danh.
        *   **Đầu ra (Output):** Tên của sinh viên được nhận diện, thời gian điểm danh (check-in), và trạng thái 'Có mặt' được cập nhật và lưu trữ."

---

## CHƯƠNG 3: CÀI ĐẶT – KIỂM THỬ – ĐÁNH GIÁ

### **Prompt 3.1: Mô hình / Thuật toán sử dụng**

**Yêu cầu:**
"Hãy viết phần **3.1: Mô hình / Thuật toán sử dụng** cho báo cáo. Tập trung hoàn toàn vào pipeline của Dlib và `face_recognition`.

Nội dung cần trình bày:
1.  **Mô hình AI chính:**
    *   Nêu rõ mô hình chính là mạng CNN dựa trên kiến trúc ResNet, được cung cấp bởi thư viện Dlib và truy cập thông qua `face_recognition`.
2.  **Pipeline xử lý:**
    *   Mô tả chi tiết các bước trong pipeline:
        *   **Face Detection:** Sử dụng phương pháp HOG (Histogram of Oriented Gradients) từ Dlib để phát hiện vị trí khuôn mặt. Có thể đề cập tùy chọn sử dụng mô hình CNN (cnn) để phát hiện chính xác hơn nhưng chậm hơn.
        *   **Face Alignment:** Căn chỉnh khuôn mặt để chuẩn hóa trước khi trích xuất đặc trưng.
        *   **Feature Extraction:** Sử dụng mô hình ResNet đã được huấn luyện sẵn để chuyển đổi khuôn mặt đã căn chỉnh thành một vector đặc trưng 128 chiều.
        *   **Face Matching:** So khớp vector mới với các vector trong cơ sở dữ liệu bằng cách tính khoảng cách Euclidean.
3.  **Lý do lựa chọn:**
    *   Giải thích tại sao mô hình này phù hợp: là một pipeline hoàn chỉnh, hiệu quả, chạy tốt trên CPU cho các ứng dụng thời gian thực, và dễ dàng cài đặt, sử dụng."

---

### **Prompt 3.2: Tập dữ liệu (Dataset)**

**Yêu cầu:**
"Hãy viết phần **3.2: Tập dữ liệu (Dataset)** cho báo cáo, dựa trên quy trình đăng ký sinh viên của hệ thống.

Nội dung cần mô tả:
1.  **Nguồn gốc và cách thu thập:**
    *   Dataset được xây dựng trong quá trình sử dụng: quản trị viên đăng ký sinh viên mới bằng cách cung cấp thông tin và ảnh chân dung qua giao diện.
2.  **Chuẩn hóa dữ liệu:**
    *   Ảnh đầu vào được hệ thống tự động xử lý: phát hiện và cắt (crop) vùng chứa khuôn mặt. Không có bước chuẩn hóa thủ công phức tạp.
3.  **Cấu trúc và yêu cầu:**
    *   Mỗi sinh viên được đăng ký với một hoặc một vài ảnh mẫu.
    *   Nêu rõ yêu cầu về chất lượng ảnh để đảm bảo độ chính xác: ảnh cần rõ nét, nhìn thẳng, đủ sáng, không bị che khuất. Có thể khuyến khích cung cấp ảnh ở nhiều góc độ khác nhau để tăng độ chính xác."

---

### **Prompt 3.3: Cấu hình hệ thống**

**Yêu cầu:**
"Hãy viết phần **3.3: Cấu hình hệ thống** cho báo cáo. Liệt kê các yêu cầu về phần mềm và phần cứng theo danh sách sau:

*   **Hệ điều hành:** Windows 10/11.
*   **Ngôn ngữ lập trình:** Python 3.8 trở lên.
*   **Các thư viện chính:**
    *   `Flask`: Web framework để xây dựng ứng dụng.
    *   `OpenCV-Python`: Để xử lý hình ảnh và video.
    *   `dlib`: Nền tảng cho các thuật toán xử lý ảnh.
    *   `face_recognition`: Thư viện cấp cao để nhận diện khuôn mặt.
    *   `Numpy`: Để xử lý các mảng và vector số.
*   **Cấu hình phần cứng đề xuất:**
    *   CPU: Intel Core i5 trở lên.
    *   RAM: Tối thiểu 8GB.
    *   Camera: Webcam tích hợp hoặc camera USB."

---

### **Prompt 3.4 & 3.5: Kết quả kiểm thử và Nhận xét - Đánh giá**

**Yêu cầu:**
"Hãy viết phần cuối của chương 3, bao gồm **3.4: Kết quả kiểm thử** và **3.5: Nhận xét – đánh giá**. Đưa ra các số liệu giả định hợp lý và bám sát vào cấu trúc sau:

1.  **Kết quả kiểm thử (Giả định):**
    *   **Tốc độ xử lý:** Trên cấu hình đề xuất, hệ thống đạt tốc độ xử lý khoảng 10-15 FPS.
    *   **Độ chính xác:** Trong điều kiện lý tưởng (ánh sáng tốt, mặt thẳng), độ chính xác đạt 95-97%.
    *   **Phân tích các tình huống thử nghiệm:**
        *   *Ánh sáng yếu:* Độ chính xác giảm đáng kể.
        *   *Góc mặt lệch:* Hệ thống vẫn có thể nhận diện ở góc nghiêng nhẹ, nhưng độ chính xác giảm khi góc lệch lớn.
        *   *Đeo khẩu trang:* Hệ thống không thể nhận diện khi khuôn mặt bị che quá nhiều.
2.  **Nhận xét và Đánh giá:**
    *   **Ưu điểm:**
        *   Thời gian xử lý tương đối nhanh, phù hợp cho ứng dụng thời gian thực trên CPU.
        *   Độ chính xác nhận diện ổn định trong điều kiện tốt.
        *   Dễ cài đặt và triển khai.
    *   **Hạn chế:**
        *   Rất nhạy cảm với điều kiện ánh sáng và góc mặt.
        *   Không hoạt động khi khuôn mặt bị che khuất (khẩu trang, kính râm).
        *   Thiếu cơ chế chống giả mạo (anti-spoofing), có thể bị đánh lừa bằng ảnh hoặc video.
    *   **Độ phức tạp thuật toán:**
        *   Mô hình CNN có số lượng tham số lớn, nhưng do sử dụng mô hình đã được huấn luyện trước (pretrained) và tối ưu hóa, nó có thể chạy hiệu quả trên CPU.
        *   Thuật toán so khớp (matching) dựa trên khoảng cách Euclidean có độ phức tạp tuyến tính O(N) (với N là số sinh viên), đáp ứng tốt cho quy mô lớp học vừa và nhỏ."
