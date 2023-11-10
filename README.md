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

Mọi người thường sử dụng các thuật ngữ “học máy”, “trí tuệ nhân tạo” và “khoa học dữ liệu” thay thế cho nhau. Trên thực tế, học máy là một tập hợp con của trí tuệ nhân tạo (AI) và có mối liên hệ chặt chẽ với khoa học dữ liệu. Trí tuệ nhân tạo xử lý bất kỳ kỹ thuật nào cho phép máy móc hiển thị “trí thông minh”, tương tự như con người. Học máy là một trong những kỹ thuật chính được sử dụng để tạo ra trí tuệ nhân tạo, nhưng các kỹ thuật không phải ML khác (ví dụ: cắt tỉa alpha-beta, hệ thống dựa trên quy tắc) cũng được sử dụng rộng rãi trong AI.

Mặt khác, khoa học dữ liệu liên quan đến việc thu thập thông tin chi tiết từ bộ dữ liệu. Theo truyền thống, các nhà khoa học dữ liệu đã sử dụng các phương pháp thống kê để thu thập những hiểu biết này. Tuy nhiên, khi học máy tiếp tục phát triển, nó cũng đã thâm nhập vào lĩnh vực khoa học dữ liệu.

Trong ngành, bất kỳ nhà khoa học dữ liệu hoặc nhà nghiên cứu AI nào cũng cần có hiểu biết tốt về học máy. Học máy trong công nghiệp đã cho phép chúng ta tạo ra các hệ thống tự động tuyệt vời. Các hệ thống này đã đạt được hoặc thậm chí đôi khi vượt quá hiệu suất tốt nhất của con người trong các lĩnh vực tương ứng của chúng. Một ví dụ điển hình là AlphaGo, một hệ thống dựa trên máy học đã đánh bại những người chơi cờ vây giỏi nhất thế giới.

### 7 Bước của Quy trình Học Máy

1. **Thu thập dữ liệu:** Quá trình trích xuất các tập dữ liệu thô cho tác vụ học máy. Dữ liệu này có thể đến từ nhiều nơi khác nhau, từ các tài nguyên trực tuyến nguồn mở đến nguồn cung ứng cộng đồng có trả phí. Bước đầu tiên của quá trình học máy được cho là quan trọng nhất. Nếu dữ liệu bạn thu thập có chất lượng kém hoặc không liên quan thì mô hình bạn đào tạo cũng sẽ có chất lượng kém.
2. **Xử lý và chuẩn bị dữ liệu:** Sau khi thu thập dữ liệu liên quan, bạn cần xử lý dữ liệu đó và đảm bảo rằng dữ liệu đó ở định dạng có thể sử dụng được để đào tạo mô hình học máy. Điều này bao gồm xử lý dữ liệu bị thiếu, xử lý các dữ liệu ngoại lệ, v.v.
3. **Kỹ thuật tính năng:** Sau khi bạn đã thu thập và xử lý tập dữ liệu của mình, bạn có thể sẽ cần phải chuyển đổi một số tính năng (và đôi khi thậm chí loại bỏ một số tính năng) để tối ưu hóa mức độ đào tạo của mô hình về dữ liệu.
4. **Lựa chọn mô hình:** Dựa trên tập dữ liệu, bạn sẽ chọn kiến ​​trúc mô hình nào sẽ sử dụng. Đây là một trong những nhiệm vụ chính của kỹ sư công nghiệp. Thay vì cố gắng đưa ra một kiến ​​trúc mô hình hoàn toàn mới, hầu hết các nhiệm vụ có thể được thực hiện triệt để với kiến ​​trúc hiện có (hoặc kết hợp các kiến ​​trúc mô hình).
5. **Đường ống dữ liệu và đào tạo mô hình:** Sau khi chọn kiến ​​trúc mô hình, bạn sẽ tạo một đường ống dữ liệu để đào tạo mô hình. Điều này có nghĩa là tạo ra một luồng quan sát dữ liệu theo đợt liên tục để huấn luyện mô hình một cách hiệu quả. Vì quá trình đào tạo có thể mất nhiều thời gian nên bạn muốn đường dẫn dữ liệu của mình hiệu quả nhất có thể.
6. **Xác thực mô hình:** Sau khi đào tạo mô hình trong một khoảng thời gian vừa đủ, bạn sẽ cần xác thực hiệu suất của mô hình trên một phần được giữ lại của tập dữ liệu tổng thể. Dữ liệu này cần phải đến từ cùng một phân phối cơ bản như tập dữ liệu huấn luyện nhưng cần phải là dữ liệu khác mà mô hình chưa từng thấy trước đây.
7. **Tính bền vững của mô hình:** Cuối cùng, sau khi đào tạo và xác nhận hiệu suất của mô hình, bạn cần có khả năng lưu đúng trọng số của mô hình và có thể đưa mô hình vào sản xuất. Điều này có nghĩa là thiết lập một quy trình mà người dùng mới có thể dễ dàng sử dụng mô hình được đào tạo trước của bạn để đưa ra dự đoán.

Sau khi tham gia khóa học này, bạn sẽ có thể thực hiện quy trình và làm sạch tập dữ liệu thô, đào tạo mô hình học máy trên dữ liệu đó và xác thực hiệu suất của mô hình. Cụ thể, bạn sẽ có thể:

- **Lấy một tập dữ liệu thô và xử lý nó cho một nhiệm vụ nhất định.** Điều này có nghĩa là xử lý dữ liệu bị thiếu và các ngoại lệ, chuẩn hóa và chuyển đổi các tính năng, tìm ra tính năng nào phù hợp nhất với nhiệm vụ và chọn ra sự kết hợp tốt nhất của các tính năng để sử dụng.
- **Chọn kiến ​​trúc mô hình chính xác để sử dụng dựa trên dữ liệu.** Nhiều người sẽ luôn mặc định sử dụng mạng thần kinh lớn cho bất kỳ tác vụ học máy nào, nhưng nhiều khi điều này là không cần thiết và thậm chí có thể ảnh hưởng đến hiệu suất cuối cùng của mô hình nếu tập dữ liệu không đủ lớn.
- **Viết mã mô hình học máy và huấn luyện nó trên dữ liệu đã được xử lý.** Xác thực hiệu suất của mô hình trên dữ liệu được cung cấp và hiểu các kỹ thuật để cải thiện hiệu suất của mô hình.

## Thao tác dữ liệu với Numpy

### Giới thiệu

#### Xử lý dữ liệu

Khi được hỏi về mô hình thành công của Google, Peter Norvig, giám đốc nghiên cứu của Google, đã có câu nói nổi tiếng:

    “Chúng tôi không có thuật toán tốt hơn bất kỳ ai khác, chúng tôi chỉ có nhiều dữ liệu hơn mà thôi.” 

Mặc dù có thể là một cách đánh giá thấp (với số lượng nhân tài được tuyển dụng tại Google), nhưng câu trích dẫn này mang lại cảm giác về tầm quan trọng của dữ liệu để đạt được kết quả thành công.

Mọi người thường thảo luận về tầm quan trọng của dữ liệu trong bối cảnh học máy. Cho dù mô hình học máy phức tạp đến đâu, nó cũng sẽ không hoạt động tốt trừ khi có lượng dữ liệu hợp lý để đào tạo. Mặt khác, với một tập dữ liệu huấn luyện lớn và đa dạng, một mô hình học sâu tốt sẽ hoạt động tốt hơn đáng kể so với các thuật toán không học sâu.

Tuy nhiên, dữ liệu không chỉ giới hạn ở việc học máy. Các công ty sử dụng dữ liệu để xác định xu hướng khách hàng, các đảng chính trị sử dụng dữ liệu để xác định nhân khẩu học nào họ nên nhắm mục tiêu, các đội thể thao sử dụng dữ liệu để phân tích người chơi, v.v.

|                                                        ![Alt text](image.png)                                                         |
| :-----------------------------------------------------------------------------------------------------------------------------------: |
| Dữ liệu ví dụ về bóng chày được sử dụng trong sabermetrics . Khái niệm này đã được phổ biến rộng rãi nhờ bộ phim Moneyball năm 2011 . |

Việc sử dụng dữ liệu phổ biến khiến cho việc xử lý dữ liệu , hành động chuyển đổi dữ liệu thô thành dạng có ý nghĩa, trở thành một kỹ năng cần thiết cần phải có.

#### NumPy

Nhiều kịch bản liên quan đến chủ yếu là các tập dữ liệu số. Ví dụ: dữ liệu y tế chứa nhiều số liệu, chẳng hạn như chiều cao, cân nặng và huyết áp. Hơn nữa, phần lớn các mạng thần kinh sử dụng dữ liệu đầu vào là số hoặc đã được chuyển đổi sang dạng số.

Khi chúng ta xử lý dữ liệu số, thư viện Python tốt nhất nên sử dụng là NumPy . Thư viện NumPy cho phép chúng ta thực hiện nhiều thao tác trên dữ liệu số và chuyển đổi dữ liệu sang các dạng dễ sử dụng hơn.

```python
import numpy as np  # import the NumPy library

# Initializing a NumPy array
arr = np.array([-1, 2, 5], dtype=np.float32)

# Print the representation of the array
print(repr(arr))
```

**Output:**

```output
array([-1.,  2.,  5.], dtype=float32)
```

### Mảng Numpy

#### Arrays

Mảng NumPy về cơ bản chỉ là danh sách Python với các tính năng bổ sung. Trên thực tế, bạn có thể dễ dàng chuyển đổi danh sách Python thành mảng Numpy bằng cách sử dụng hàm ***np.array*** lấy danh sách Python làm đối số bắt buộc. Hàm này cũng có khá nhiều đối số từ khóa, nhưng đối số chính cần biết là ***dtype***. Đối số từ khóa ***dtype*** lấy type NumPy và chuyển mảng theo cách thủ công sang type đã chỉ định.

Đoạn mã dưới đây là một ví dụ về cách sử dụng np.arrayđể tạo ma trận 2-D. Lưu ý rằng mảng được truyền thủ công tới ***np.float32.***

```python
import numpy as np

arr = np.array([[0, 1, 2], [3, 4, 5]],
               dtype=np.float32)
print(repr(arr))
```

**Output:**

```output
array([[0., 1., 2.],
       [3., 4., 5.]], dtype=float32)
```

Khi các phần tử của mảng NumPy là các kiểu hỗn hợp thì loại của mảng sẽ được nâng cấp lên loại cấp cao nhất. Điều này có nghĩa là nếu đầu vào mảng có hỗn hợp các phần tử int và float, tất cả các số nguyên sẽ được chuyển sang giá trị tương đương với dấu phẩy động của chúng. Nếu một mảng được trộn lẫn các phần tử int, float, và string, mọi thứ đều được chuyển thành chuỗi.

Mã dưới đây là một ví dụ về np.array upcasting. Cả hai số nguyên đều được chuyển sang dạng dấu phẩy động tương đương của chúng.

```python
arr = np.array([0, 0.1, 2])
print(repr(arr))
```

**Output:**

```output
array([0. , 0.1, 2. ])
```

#### Copying

Tương tự như danh sách Python, khi chúng ta tạo tham chiếu đến mảng NumPy, nó không tạo ra một mảng khác. Do đó, nếu chúng ta thay đổi một giá trị bằng biến tham chiếu, nó cũng sẽ thay đổi mảng ban đầu. Chúng ta giải quyết vấn đề này bằng cách sử dụng chức năng copy() vốn có của mảng Numpy. Hàm không có đối số bắt buộc và nó trả về mảng đã sao chép.

Trong ví dụ mã bên dưới, c là một tham chiếu đến a, trong khi d là một bản sao. Vì thế, việc thay đổi c dẫn đến sự thay đổi tương tự trong a, trong khi thay đổi d không làm thay đổi giá trị của b.

```python
a = np.array([0, 1])
b = np.array([9, 8])
c = a
print('Array a: {}'.format(repr(a)))
c[0] = 5
print('Array a: {}'.format(repr(a)))

d = b.copy()
d[0] = 6
print('Array b: {}'.format(repr(b)))
```

**Output:**

```output
Array a: array([0, 1])
Array a: array([5, 1])
Array b: array([9, 8])
```

#### Casting

Chúng ta ép kiểu mảng Numpy thông qua hàm astype() của nó. Đối số bắt buộc của hàm là kiểu mới cho mảng. Nó trả về mảng đã chuyển thành kiểu mới.

Mã bên dưới hiển thị một ví dụ về việc ép kiểu bằng cách sử dụng hàm astype(). Thuộc tính dtype trả về kiểu của mảng.

```python
arr = np.array([0, 1, 2])
print(arr.dtype)
arr = arr.astype(np.float32)
print(arr.dtype)
```

**Output:**

```output
int64
float32
```

#### NaN

Khi chúng ta không muốn mảng NumPy chứa một giá trị tại một chỉ mục cụ thể, chúng ta có thể sử dụng np.***nan*** để đóng vai trò giữ chỗ. Một cách sử dụng phổ biến cho ***np.nan*** là giá trị điền cho dữ liệu không đầy đủ.

Đoạn mã dưới đây cho thấy một ví dụ về cách sử dụng ***np.nan***. Lưu ý rằng ***np.nan*** không thể đảm nhận một loại số nguyên.

```python
arr = np.array([np.nan, 1, 2])
print(repr(arr))

arr = np.array([np.nan, 'abc'])
print(repr(arr))

# Will result in a ValueError: If we uncomment line 8 and run again.
#np.array([np.nan, 1, 2], dtype=np.int32)
np.array([np.nan, 1, 2], dtype=np.float32)
```

**Output:**

```output
array([nan,  1.,  2.])
array(['nan', 'abc'], dtype='<U32')
```

#### Infinity

Để biểu thị vô cực trong NumPy, chúng ta sử dụng giá trị đặc biệt ***np.inf***. Chúng ta cũng có thể biểu diễn âm vô cực bằng ***-np.inf***.

Đoạn mã dưới đây cho thấy một ví dụ về cách sử dụng ***np.inf***. Lưu ý rằng ***np.inf*** không thể đảm nhận một loại số nguyên.

```python
print(np.inf > 1000000)

arr = np.array([np.inf, 5])
print(repr(arr))

arr = np.array([-np.inf, 1])
print(repr(arr))

# Will result in a OverflowError: If we uncomment line 10 and run again.
# np.array([np.inf, 3], dtype=np.int32)
np.array([np.inf, 3], dtype=np.float32)
```

**Output:**

```output
True
array([inf,  5.])
array([-inf,   1.])
```

### Khái niệm cơ bản về Numpy

### Toán học

### Ngẫu nhiên

### Lập chỉ mục

### Lọc

### Số liệu thống kê

### Tổng hợp

### Lưu dữ liệu
