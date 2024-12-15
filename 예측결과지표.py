import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 예시 데이터 (실제 각도와 예측된 각도)
actual_angles = [30, 60, 90, 120, 150]  # 실제 각도 데이터
predicted_angles = [32, 58, 88, 130, 145]  # 모델 예측 결과

# MSE 계산
mse = mean_squared_error(actual_angles, predicted_angles)
print(f'Mean Squared Error: {mse:.4f}')

# R² Score 계산
r2 = r2_score(actual_angles, predicted_angles)
print(f'R² Score: {r2:.4f}')

# 산점도 시각화
plt.scatter(actual_angles, predicted_angles, label='Predicted Angles', color='blue')

# y = x 선 (완벽한 예측을 나타냄)
plt.plot([min(actual_angles), max(actual_angles)], 
         [min(actual_angles), max(actual_angles)], 
         color='red', label='Perfect Prediction (y=x)')

# 그래프 축 레이블 및 제목 설정
plt.xlabel('Actual Angles')
plt.ylabel('Predicted Angles')
plt.title('Actual vs Predicted Angles')

# 범례 추가
plt.legend()

# 그래프 표시
plt.show()
