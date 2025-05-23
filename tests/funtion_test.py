import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成示例二值图像（含噪声）
def create_sample_image():
    img = np.zeros((200, 400), dtype=np.uint8)
    # 添加两个白色矩形（模拟物体）
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    cv2.rectangle(img, (200, 80), (300, 180), 255, -1)
    # 添加孤立噪声点
    cv2.circle(img, (30, 30), 5, 255, -1)
    cv2.circle(img, (350, 120), 3, 255, -1)
    return img

# 2. 定义腐蚀函数
def apply_erosion(img, kernel, iterations):
    eroded = cv2.erode(img, kernel, iterations=iterations)
    return eroded

# 3. 可视化对比
def visualize(original, processed, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(original, cmap='gray')
    plt.title('Original'), plt.axis('off')
    plt.subplot(122), plt.imshow(processed, cmap='gray')
    plt.title(title), plt.axis('off')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 生成样本图像
    img = create_sample_image()
    
    # 定义腐蚀参数
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 3x3矩形核
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 5x5十字核
    
    # 应用腐蚀（不同参数）
    eroded_rect_1 = apply_erosion(img, kernel_rect, iterations=1)
    eroded_rect_2 = apply_erosion(img, kernel_rect, iterations=2)
    eroded_cross_1 = apply_erosion(img, kernel_cross, iterations=1)
    
    # 可视化结果
    visualize(img, eroded_rect_1, 'Erosion (3x3 Rect, 1 Iter)')
    visualize(img, eroded_rect_2, 'Erosion (3x3 Rect, 2 Iters)')
    visualize(img, eroded_cross_1, 'Erosion (5x5 Cross, 1 Iter)')