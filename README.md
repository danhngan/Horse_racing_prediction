# **Set up horse race environment**


# **Step 1: Business understanding**
## **Sơ qua về hệ thống cá cược đua ngựa tại Nhật Bản** : Bạn có thể xem [tại đây](https://japanracing.jp/en/racing/go_racing/guide/)
- Gồm 10 trường đua.
- Có 12 cuộc đua trong một ngày thi đấu, thường bắt đầu từ 10:00.
- Ngoài các cuộc đua thông thường, còn có các cuộc đua trong các hệ thống giải.
- Các cuộc đua có thể có các điều kiện riêng, ví dụ cuộc đua phân loại theo tuổi ngựa, giới tính ngựa,...
- Số lượng ngựa đua trong một cuộc đua là không cố định

## **Mục tiêu dự án**
Ở đây tôi muốn dự đoán **top 3** con ngựa trong mỗi cuộc đua

**Metrics:** `accuracy`

**Kết quả mong muốn:** `>=49.3%`

Sẽ có một số cách tiếp cận sau:
1. Dự đoán khả năng dàng top 1 của tất cả các con ngựa trong đua và chọn ra 3 con có khả năng cao nhất.
2. Dự đoán trực tiếp khả năng vào top 3 (có thể chọn top n) của tất cả các con ngựa trong cuộc đua và chọn ra 3 con có khả năng cao nhất.
3. Dự đoán thời gian chạy của mỗi con ngựa trong cuộc đua và chọn ra 3 con nhanh nhất.

Mỗi phương pháp có thể có một số ưu nhược điểm nhất định:
1. Phương pháp 1 và 2:
   1. Ưu điểm: Trực tiếp dự đoán được top các con ngựa, không cần post-processing. Thể hiện được tính cạnh tranh giữa các con ngựa.
   2. Nhược điểm: Khó khăn trong pre-processing khi số lượng ngựa trong một cuộc đua không cố định, các đặc tính chung giữa các con ngựa trong cuộc đua trùng lặp như điều kiện thời tiết, đường đua. Framework như sklearn không hỗ trợ thuộc tính là ma trận 2 chiều, việc flatten dữ liệu có thể  làm cho đầu vào có quá nhiều thuộc tính.
2. Phương pháp 3:
   1. Ưu điểm: Preprocessing dễ dàng hơn khi chỉ cần quan tâm đến các yếu tố ảnh hưởng cho từng con ngựa một, đầu vào cho mỗi điểm dữ liệu chỉ là một vector thể hiện cho 1 con ngựa trong cuộc đua.
   2. Nhược điểm: Không thể hiện tính cạnh tranh giữa các con ngựa trong cuộc đua, cần post-proccessing để đưa ra dự đoán cuối cùng

Ở đây, đầu tiên tôi sẽ chọn cách tiếp cận 3, sau đó sẽ thử nghiệm tiếp tục với cách tiếp cận 1 và 2

Tuy vậy, dù theo cách tiếp cận nào ta cũng cần biết những yếu tố nào ảnh hưởng đến kết quả của một con ngựa trong cuộc đua:

# **Step 2 Data requirement**

Một số yếu tố có thể ảnh hưởng đến kết quả cuộc đua:
- Về ngựa: khối lượng, tuổi, giống ngựa, giới tính, các thành tích đã đạt được
- Về người điều khiển: kinh nghiệm thi đấu, các thành tích
- Ngoại cảnh: điều kiện thời tiết, điều kiện trường đua, loại đường đua
- Các ngựa đối thủ, đồng đội trong cuộc đua

Sơ lược về data của chúng ta:
- Có 9 tập dữ liệu cho các cuộc đua cũ và 10 tập dữ liệu cho các cuộc đua gần đây:
  - RA: Race details
  - SE: Horse information in each race
  - UM: Horse detail information
  - KS: Jockey master
  - CH: Master of training
  - BR: Product master
  - BN: Owner master
  - HN: Breeding horse
  - WH: Horse weight (use for new race)
  - WE: Weather condition
- Trong đó hai tập dữ liệu RA và SE chứa nhiều thông tin quan trọng, bạn có thể xem mô tả cụ thể trong file `Horse-Race-Data-Description.xlsx`
- Nhận xét, bộ data đã có khá đầy đủ các yêu cầu cuẩ chúng ta.

# Step 3: Data cleaning và EDA

Trước khi đi vào phân tích data, chúng ta sẽ loại bỏ một số thuộc tính không hữu ích, đồng thời tổng hợp data từ các file dữ liệu để thuận tiện hơn sau này

Xem notebook `firstlooking.ipynb` cho phần này. Tiêu chí loại ban đầu (Chưa chú trọng đến mức độ hiệu quả cho việc dự đoán):
- Data có kiểu dữ liệu object trong pandas (kí tự) mà có quá nhiều giá trị (ngoại trừ thông tin định danh jockey và training master phục vụ cho việc phân tích theo cấp độ từng con ngựa hoặc từng jockey)
- Các trường dữ liệu chỉ có một giá trị (không có tính phân loại)
- Trường có ý nghĩa trùng lặp
- Các trường chỉ có sau khi đã bắt đầu cuộc đua (trừ các trường mục tiêu là thứ tự xếp hạng và thời gian)
- Tổng hợp các thông tin từ các file dữ liệu
  
Xem notebook `cleaning_EDA.ipynb` cho phần cleaning.
- Xử lý cách dữ liệu thiếu
- Xử lý data type
- Loại bỏ thêm một số trường không phù hợp
- Xem xét tính đúng đắn một số trường
- Phân tích khả năng dự đoán của dữ liệu
- Phân tích mối tương quan của một số thuộc tính
- Trực quan hóa các phân tích
- Loại bỏ một số trường có ít không có khả năng dự đoán
- Đưa ra bản rút gọn phù hợp của data

Cần chú ý rằng:

Khi chúng ta phân tích theo cấp độ ngựa hay cấp độ cuộc đua, chúng ta muốn biết rằng giới tính ngựa nào có kết quả tốt hơn, giống thuần chủng hay lai tốt hơn. Tuy nhiên vấn đề gặp phải là một con ngựa có thể đua nhiều cuộc đua, mỗi cuộc đua có thể chỉ có một số loại ngựa tham gia (do quy định hoặc tự nguyện), và số lượng ngựa ở từng nhóm cũng có khác biệt lớn. Điều này dẫn đến việc, nếu ra tổng hợp theo từng con ngựa, kết quả thu được có thể bị không phản ánh đúng. Việc tổng hợp kết quả theo từng cuộc đua cũng có thể  không cho kết quả tốt, vì những con ngựa tốt thường được dùng thường xuyên, làm ta nghĩ rằng cả nhóm đó là tốt.

Ở đây, tôi đề xuất một cách để so sánh, đó chính là so sánh các nhóm ngựa dựa trên các cuộc đua có sự tham gia của các nhóm cần so sánh (hoặc dựa trên các điều kiện đua tương đương) và chỉ lấy kết quả trung bình của từng con, có thể loại bỏ một số con ngựa có ít dữ liệu. Nhược điểm là vẫn có thể bỏ sót sự đóng góp của các con ngựa mà không có dữ liệu có thể so sánh
