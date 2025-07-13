#!/usr/bin/env python
# Created by "Thieu" at 13:57, 13/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd

# Đọc file CSV
df = pd.read_csv('superconductivty_full.csv')

# Kiểm tra kích thước ban đầu
print("Original shape:", df.shape)


# Lấy mẫu ngẫu nhiên không lặp
df_sampled = df.sample(n=7500, random_state=42)  # random_state để tái tạo kết quả

# Kiểm tra kích thước sau khi lấy mẫu
print("Sampled shape:", df_sampled.shape)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Phân bố trước khi rút gọn
plt.subplot(1, 2, 1)
df['critical_temp'].hist(bins=50, color='skyblue', edgecolor='black')
plt.title('Original critical_temp distribution')
plt.xlabel('Critical Temperature')
plt.ylabel('Frequency')

# Phân bố sau khi rút gọn
plt.subplot(1, 2, 2)
df_sampled['critical_temp'].hist(bins=50, color='salmon', edgecolor='black')
plt.title('Sampled (7500) critical_temp distribution')
plt.xlabel('Critical Temperature')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

df_sampled.to_csv('superconductivity.csv', index=False)
