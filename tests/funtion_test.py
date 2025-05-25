import cv2
import numpy as np
import matplotlib.pyplot as plt

def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region.astype(np.uint8)

# 创建一个示例二值掩码
mask = np.zeros((200, 200), dtype=np.uint8)

# 添加多个连通区域
cv2.rectangle(mask, (50, 50), (150, 150), 1, -1)  # 大矩形
cv2.circle(mask, (100, 100), 40, 1, -1)          # 中等圆形
cv2.rectangle(mask, (30, 30), (80, 80), 1, -1)   # 小矩形
cv2.circle(mask, (170, 170), 20, 1, -1)          # 小圆形

# 应用 select_max_region 函数
max_region = select_max_region(mask)

# 可视化结果
plt.figure(figsize=(12, 6))

# 原始掩码
plt.subplot(1, 2, 1)
plt.title("Original Mask")
plt.imshow(mask, cmap='gray')
plt.axis('off')

# 最大连通区域
plt.subplot(1, 2, 2)
plt.title("Max Region")
plt.imshow(max_region, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()