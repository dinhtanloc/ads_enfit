# Mô tả Bộ Dữ Liệu
Thử thách của bạn trong cuộc thi này là dự đoán lượng điện được sản xuất và tiêu thụ bởi các khách hàng năng lượng ở Estonia đã lắp đặt các tấm pin mặt trời. Bạn sẽ có quyền truy cập vào dữ liệu thời tiết, giá năng lượng liên quan và hồ sơ về công suất quang điện được lắp đặt.

Đây là cuộc thi dự báo sử dụng API chuỗi thời gian. Bảng xếp hạng riêng tư sẽ được xác định bằng dữ liệu thực tế thu thập sau khi kết thúc thời gian nộp bài. Bộ dữ liệu có thể được lấy tại đây: https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers

💡 Lưu ý: Tất cả các bộ dữ liệu tuân theo cùng một quy ước thời gian. Thời gian được cung cấp theo EET/EEST. Hầu hết các biến là tổng hoặc trung bình trong khoảng thời gian 1 giờ. Cột datetime (dù tên là gì) luôn cho biết thời điểm bắt đầu của khoảng thời gian 1 giờ. Tuy nhiên, đối với các bộ dữ liệu thời tiết, một số biến như nhiệt độ hoặc độ che phủ mây được cung cấp cho một thời điểm cụ thể, luôn là thời điểm kết thúc của khoảng thời gian 1 giờ.

#### Các tệp train.csv

county: Mã ID cho huyện.
is_business: Boolean cho biết liệu người tiêu thụ có phải là doanh nghiệp hay không.
product_type: Mã ID với ánh xạ sau đây của các mã tới các loại hợp đồng: {0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot"}.
target: Lượng tiêu thụ hoặc sản xuất cho phân đoạn liên quan trong giờ. Các phân đoạn được xác định bởi county, is_business, và product_type.
is_consumption: Boolean cho biết liệu mục tiêu của hàng này là tiêu thụ hay sản xuất.
datetime: Thời gian Estonia theo EET (UTC+2) / EEST (UTC+3). Nó mô tả thời điểm bắt đầu của khoảng thời gian 1 giờ mà mục tiêu được cung cấp.
data_block_id: Tất cả các hàng chia sẻ cùng một data_block_id sẽ có sẵn tại cùng thời điểm dự báo. Đây là chức năng của những thông tin có sẵn khi các dự báo thực sự được thực hiện, vào lúc 11 giờ sáng mỗi ngày. Ví dụ, nếu dự báo thời tiết data_block_id cho các dự báo được thực hiện vào ngày 31 tháng 10 là 100 thì data_block_id thời tiết lịch sử cho ngày 31 tháng 10 sẽ là 101 vì dữ liệu thời tiết lịch sử chỉ thực sự có sẵn vào ngày hôm sau.
row_id: Một mã định danh duy nhất cho hàng.
prediction_unit_id: Một mã định danh duy nhất cho sự kết hợp giữa county, is_business, và product_type. Các đơn vị dự báo mới có thể xuất hiện hoặc biến mất trong tập kiểm tra.

#### gas_prices.csv

origin_date: Ngày khi giá trước một ngày có sẵn.
forecast_date: Ngày khi giá dự báo có liên quan.
[lowest/highest]_price_per_mwh: Giá thấp nhất/cao nhất của khí đốt tự nhiên trên thị trường trước một ngày vào ngày giao dịch đó, tính bằng Euro mỗi megawatt giờ tương đương.
data_block_id

#### client.csv

product_type
county: Mã ID cho huyện. Xem county_id_to_name_map.json để ánh xạ mã ID tới tên huyện.
eic_count: Số điểm tiêu thụ tổng hợp (EICs - Mã định danh Châu Âu).
installed_capacity: Công suất tấm pin mặt trời quang điện được lắp đặt tính bằng kilowatt.
is_business: Boolean cho biết liệu người tiêu thụ có phải là doanh nghiệp hay không.
date
data_block_id

#### electricity_prices.csv

origin_date
forecast_date: Đại diện cho thời điểm bắt đầu của khoảng thời gian 1 giờ khi giá có hiệu lực.
euros_per_mwh: Giá điện trên các thị trường trước một ngày tính bằng Euro mỗi megawatt giờ.
data_block_id

#### forecast_weather.csv

Dự báo thời tiết có sẵn vào thời điểm dự báo. Nguồn từ Trung tâm Dự báo Thời tiết Tầm trung Châu Âu.

[latitude/longitude]: Tọa độ của dự báo thời tiết.
origin_datetime: Dấu thời gian của khi dự báo được tạo ra.
hours_ahead: Số giờ giữa thời điểm tạo dự báo và thời điểm dự báo thời tiết. Mỗi dự báo bao gồm tổng cộng 48 giờ.
temperature: Nhiệt độ không khí ở 2 mét trên mặt đất tính bằng độ C. Được ước tính cho thời điểm kết thúc của khoảng thời gian 1 giờ.
dewpoint: Nhiệt độ điểm sương ở 2 mét trên mặt đất tính bằng độ C. Được ước tính cho thời điểm kết thúc của khoảng thời gian 1 giờ.
cloudcover_[low/mid/high/total]: Tỷ lệ phần trăm của bầu trời được che phủ bởi mây ở các dải độ cao sau: 0-2 km, 2-6 km, 6+ km, và tổng cộng. Được ước tính cho thời điểm kết thúc của khoảng thời gian 1 giờ.
10_metre_[u/v]_wind_component: Thành phần [hướng đông/hướng bắc] của tốc độ gió đo ở 10 mét trên bề mặt tính bằng mét trên giây. Được ước tính cho thời điểm kết thúc của khoảng thời gian 1 giờ.
data_block_id
forecast_datetime: Dấu thời gian của thời tiết dự báo. Được tạo từ origin_datetime cộng với hours_ahead. Điều này đại diện cho thời điểm bắt đầu của khoảng thời gian 1 giờ mà dữ liệu thời tiết được dự báo.
direct_solar_radiation: Bức xạ mặt trời trực tiếp chiếu xuống bề mặt trên một mặt phẳng vuông góc với hướng của mặt trời tích lũy trong giờ, tính bằng watt-giờ trên mét vuông.
surface_solar_radiation_downwards: Bức xạ mặt trời, bao gồm cả trực tiếp và khuếch tán, chiếu xuống một mặt phẳng ngang trên bề mặt Trái Đất, tích lũy trong giờ, tính bằng watt-giờ trên mét vuông.
snowfall: Lượng tuyết rơi trong giờ tính bằng mét nước tương đương.
total_precipitation: Lượng mưa tích lũy, bao gồm cả mưa và tuyết rơi trên bề mặt Trái Đất trong giờ được mô tả, tính bằng mét.

#### historical_weather.csv

Dữ liệu thời tiết lịch sử.

datetime: Đại diện cho thời điểm bắt đầu của khoảng thời gian 1 giờ mà dữ liệu thời tiết được đo.
temperature: Được đo ở thời điểm kết thúc của khoảng thời gian 1 giờ.
dewpoint: Được đo ở thời điểm kết thúc của khoảng thời gian 1 giờ.
rain: Khác với các quy ước dự báo. Lượng mưa từ các hệ thống thời tiết lớn trong giờ tính bằng milimet.
snowfall: Khác với các quy ước dự báo. Lượng tuyết rơi trong giờ tính bằng centimet.
surface_pressure: Áp suất không khí ở mặt đất tính bằng hectopascal.
cloudcover_[low/mid/high/total]: Khác với các quy ước dự báo. Độ che phủ mây ở các độ cao 0-3 km, 3-8 km, 8+, và tổng cộng.
windspeed_10m: Khác với các quy ước dự báo. Tốc độ gió ở 10 mét trên mặt đất tính bằng mét trên giây.
winddirection_10m: Khác với các quy ước dự báo. Hướng gió ở 10 mét trên mặt đất tính bằng độ.
shortwave_radiation: Khác với các quy ước dự báo. Bức xạ toàn cầu trên mặt phẳng ngang tính bằng watt-giờ trên mét vuông.
direct_solar_radiation
diffuse_radiation: Khác với các quy ước dự báo. Bức xạ khuếch tán tính bằng watt-giờ trên mét vuông.
[latitude/longitude]: Tọa độ của trạm thời tiết.
data_block_id

#### public_timeseries_testing_util.py

Một tệp tùy chọn nhằm giúp dễ dàng hơn khi chạy các bài kiểm tra API tùy chỉnh offline. Xem docstring của script để biết chi tiết. Bạn sẽ cần chỉnh sửa tệp này trước khi sử dụng.

#### example_test_files/

Dữ liệu nhằm minh họa cách API hoạt động. Bao gồm các tệp và cột tương tự được cung cấp bởi API. Ba data_block_id đầu tiên là sự lặp lại của ba data_block_id cuối cùng trong tập huấn luyện.

#### example_test_files/sample_submission.csv

Một mẫu nộp hợp lệ, được cung cấp bởi API. Xem notebook này để biết một ví dụ rất đơn giản về cách sử dụng mẫu nộp.

#### example_test_files/revealed_targets.csv

Các giá trị mục tiêu thực tế từ ngày trước thời gian dự báo. Điều này tương đương với độ trễ hai ngày so với thời gian dự báo trong test.csv.

#### enefit/

Các tệp cho phép API hoạt động. Dự kiến API sẽ cung cấp tất cả các hàng trong vòng dưới 15 phút và chỉ sử dụng ít hơn 0,5

GB bộ nhớ. Phiên bản API mà bạn có thể tải xuống cung cấp dữ liệu từ example_test_files/. Bạn phải đưa ra dự báo cho những ngày đó để tiến hành API nhưng những dự báo đó sẽ không được chấm điểm. Dự kiến có khoảng ba tháng dữ liệu được cung cấp ban đầu và tối đa mười tháng dữ liệu vào cuối kỳ dự báo.
