# Học máy với Python

## Tổng quan

Nếu bạn là kỹ sư phần mềm đang muốn bổ sung khả năng học máy vào bộ kỹ năng của mình thì đây là nơi để bắt đầu.

Khóa học này sẽ dạy bạn viết mã hữu ích và tạo các ứng dụng máy học có tác động ngay lập tức. Ngay từ đầu, bạn sẽ được cung cấp tất cả các công cụ cần thiết để tạo các dự án machine learning cấp ngành. Thay vì đọc qua lý thuyết dày đặc, bạn sẽ học các kỹ năng thực tế và đạt được những hiểu biết sâu sắc có thể áp dụng được. Các chủ đề được đề cập bao gồm phân tích/trực quan hóa dữ liệu, kỹ thuật tính năng, học có giám sát, học không giám sát và học sâu. Tất cả các chủ đề này đều được dạy bằng các khung tiêu chuẩn ngành: NumPy, pandas, scikit-learn, XGBoost, TensorFlow và Keras.

Kiến thức cơ bản về Python là điều kiện tiên quyết cho khóa học này.

### Học máy là gì?

Học máy (ML - Machine Learning) là nhánh khoa học liên quan đến các thuật toán và hệ thống thực hiện các nhiệm vụ cụ thể bằng cách sử dụng các mẫu và suy luận, thay vì các hướng dẫn được lập trình rõ ràng. Có nhiều trường hợp sử dụng khác nhau cho học máy, từ nhận dạng hình ảnh đến tạo văn bản. Hầu hết các nhiệm vụ học máy đều khái quát hóa thành một trong hai loại học tập sau:

- **Học có giám sát:** Sử dụng dữ liệu được dán nhãn để huấn luyện mô hình. Các nhãn cho tập dữ liệu huấn luyện thể hiện lớp/danh mục mà mỗi quan sát dữ liệu thuộc về. Sau khi đào tạo, mô hình sẽ có thể dự đoán nhãn cho các quan sát dữ liệu mới (từ cùng phân bố dân số với dữ liệu huấn luyện).
  - Ví dụ: Giả sử bạn đang huấn luyện một mô hình học máy để dự đoán liệu một bức ảnh có chứa hồ hay không. Với học có giám sát, bạn sẽ huấn luyện một mô hình trên một tập dữ liệu gồm các ảnh trong đó nhãn cho mỗi ảnh là “Có” nếu nó chứa hồ hoặc “Không” nếu không có hồ. Sau khi đào tạo, mô hình sẽ có thể chụp ảnh và xác định xem nó có chứa hồ hay không.
- **Học không giám sát:** Sử dụng dữ liệu không được gắn nhãn để cho phép mô hình tìm hiểu mối quan hệ giữa các quan sát dữ liệu và chọn ra các mẫu cơ bản. Hầu hết dữ liệu trên thế giới đều không được gắn nhãn, điều này khiến cho việc học không giám sát trở thành một phương pháp học máy rất hữu ích.
  - Ví dụ: Quay lại tập dữ liệu hình ảnh tương tự ở trên, nhưng bây giờ giả sử tập dữ liệu huấn luyện không được gắn nhãn. Sử dụng phương pháp học không giám sát, một mô hình sẽ có thể nhận ra sự khác biệt vốn có giữa ảnh có hồ và ảnh không có hồ, ví dụ: sự khác biệt về màu sắc hoặc hướng pixel. Điều này cho phép mô hình phân cụm các hình ảnh thành hai nhóm riêng biệt.

Nếu có thể có được các tập dữ liệu huấn luyện được gắn nhãn đủ lớn thì học có giám sát là cách tốt nhất. Tuy nhiên, thường rất khó để có được các bộ dữ liệu được dán nhãn đầy đủ, đó là lý do tại sao nhiều nhiệm vụ yêu cầu học không giám sát hoặc học bán giám sát (sự kết hợp giữa học có giám sát và không giám sát). Quyết định sử dụng loại phương pháp học nào chỉ là bước đầu tiên để tạo ra mô hình học máy. Bạn cũng cần chọn kiến ​​trúc mô hình phù hợp cho nhiệm vụ của mình và quan trọng nhất là có thể xử lý dữ liệu thành quy trình đào tạo và diễn giải/phân tích kết quả mô hình.

### Machine Learning vs Artificial Intelligence vs Data Science
