
import streamlit as st

def render():
    st.header("About this Project")
    st.markdown(
        """
        🎯 Mục tiêu dự án

            Phân tích bộ dữ liệu Pima Indians Diabetes để hiểu rõ đặc điểm của những người có/nguy cơ mắc bệnh tiểu đường.
            Khảo sát dữ liệu (EDA) nhằm phát hiện các giá trị bất thường (như 0 ở cột Glucose, BloodPressure, …).
            Trực quan hoá để quan sát phân phối đặc trưng và mối tương quan giữa các biến.
            Xây dựng và đánh giá các mô hình Machine Learning giúp dự đoán khả năng mắc tiểu đường.
            Tìm hiểu ảnh hưởng của cân bằng lớp (SMOTE), Ensemble (Voting/Stacking) và Hyperparameter Search đến hiệu năng.

        ⚙️ Chức năng chính của ứng dụng

            Data Exploration
            Hiển thị dữ liệu thô, kiểu dữ liệu, thống kê mô tả.
            Đếm giá trị thiếu và số lượng giá trị 0 bất thường ở các cột Pima.
            Visualization
            Vẽ histogram và boxplot cho từng đặc trưng số.
            Vẽ correlation heatmap để quan sát quan hệ giữa các biến.
            Prediction (Modeling)
            Chọn thuật toán (Logistic Regression, Random Forest, SVC, XGBoost, LightGBM, Voting/Stacking).
            Tuỳ chọn xử lý dữ liệu: SMOTE, chuẩn hoá đặc trưng.
            Hỗ trợ tìm kiếm siêu tham số (GridSearch, RandomizedSearch).
            Đánh giá bằng Accuracy, Precision, Recall, F1, ROC-AUC.
            Hiển thị Confusion Matrix, ROC curve.
            Form nhập tay để dự đoán cho một bệnh nhân mới.
        """
    )
