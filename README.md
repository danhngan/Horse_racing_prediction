# Dự đoán kết quả đua ngựa Nhật Bản
<div style="text-align: right"><em>Tác giả: Trần Danh Ngân</em></div>

<div style="text-align: right"><em>email: danhnganc3@gmail.com</em></div>

<div style="text-align: right"><a href="https://www.linkedin.com/in/danhnganc3/"><em>linkedin</em></a></div>

## **Sử dụng API**
### Yêu cầu:
- Cài đặt anaconda với python version >= 3.8
- fastapi==0.78.0
- tensorflow==2.8.0
- xgboost==1.6.1
- uvicorn==0.18.3

### Sử dụng:
- Có hai phương thức được cung cấp:
  - get("/descriptions/{code}"): lấy các mô tả và ví dụ dựa trên code được cung cấp. code có thể là `SexCD`, `HinsyuCD`, `TozaiCD`, `raceid`, `horses`, `jockeys`, `horseinfos` và `tracktype`
  - post("/predictor/") với các tham số truyền dưới dạng query hoặc dùng json: các tham số gồm `raceid`,`horses`,`jockeys`,`tracktype`,`horseinfos`. Nếu `raceid` được truyền, api sẽ trả về kết quả và dự đoán tương ứng với mã cuộc đua đó. Nếu không bạn phải truyền tối thiểu 2 tham số `horses`,`jockeys` hai tham số có dạng list phải có cùng độ dài. Với các con ngựa không có tron cơ sở dữ liệu, bạn có thể cung cấp thông tin thông qua tham số `horseinfos` (khuyến khích chỉ cung cấp các thông tin cơ bản). Truy cập "/descriptions/{code}" để tìm hiểu thêm về kiểu dữ liệu cũng như các ví dụ.
- Bạn có thể truy cập "/docs" hoặc mở file "api-docs.html" trong thư mục src để đọc mô tả cụ thể, và có thể chạy thử truy vấn khi truy cập "/docs"
