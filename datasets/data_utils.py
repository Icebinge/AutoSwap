import numpy as np
import torch
import cv2

def mask_score(mask):
    """
    根据连接性对掩码进行评分。
    
    参数:
        mask: 二值掩码（numpy数组）
    
    返回:
        评分值
    """
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    conc_score = np.max(cnt_area) / np.sum(cnt_area)
    return conc_score

def sobel(img, mask, thresh = 50):
    """
    计算图像的Sobel边缘检测结果，并根据掩码进行处理。
    
    参数:
        img: 输入图像（numpy数组）
        mask: 二值掩码（用于获取图像尺寸，实际未直接使用掩码内容）
        thresh: 阈值，用于Sobel边缘检测
    
    返回:
        处理后的图像
    """
    H, W = img.shape[0], img.shape[1]
    img = cv2.resize(img, (256, 256))
    mask = (cv2.resize(mask, (256, 256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)

    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = Ksize)
    sobel_X = cv2.convertScaleAbs(sobelx)
    sobel_Y = cv2.convertScaleAbs(sobely)
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr, -1) * mask

    scharr[scharr < thresh] = 0
    scharr = np.stack([scharr] * 3, axis = -1)
    scharr = (scharr.astype(np.float32) / 255.0 * img.astype(np.float32)).astype(np.uint8)
    scharr = cv2.resize(scharr, (W, H))
    return scharr

def resize_and_pad(image, box):
    """
    将图像调整到，同时保持纵横比。
    
    参数:
        image: 输入图像（numpy数组）
        box: 边界框坐标 (y1, y2, x1, x2)
    
    返回:
        调整后的图像
    """
    y1, y2, x1, x2 = box
    H, W = y2 - y1, x2 - x1
    h, w = image.shape[0], image.shape[1]
    r_box = W / H
    r_image = w / h
    if r_box >= r_image:
        h_target = H
        w_target = int(H * r_image)
        image = cv2.resize(image, (w_target, h_target))

        w1 = (W - w_target) // 2
        w2 = W - w_target - w1
        image = np.pad(image, ((0, 0), (w1, w2), (0, 0)), mode = 'constant', constant_values = 255)
    else:
        w_target = W
        h_target = int(W * r_image)
        image = cv2.resize(image, (w_target, h_target))

        h1 = (H - h_target) // 2
        h2 = H - h_target - h1
        image = np.pad(image, ((h1, h2), (0, 0), (0, 0)), mode = 'constant', constant_values = 255)
    return image

def expand_image_mask(image, mask, ratio=1.4):
    """
    扩展图片和掩码。
    
    参数:
        image: 输入图像（numpy数组）
        mask: 二值掩码（numpy数组）
        ratio: 扩展比例
    
    返回:
        扩展后的图像和掩码
    """
    h, w = image.shape[0], image.shape[1]
    H, W = int(h * ratio), int(w * ratio)
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W - w) // 2)
    w2 = W - w - w1
    image = np.pad(image, ((h1, h2), (w1, w2), (0, 0)), mode = 'constant', constant_values = 255)
    mask = np.pad(mask, ((h1, h2), (w1, w2)), mode = 'constant', constant_values = 0)
    return image, mask

def resize_box(yyxx, H, W, h, w):
    """
    调整框的大小。
    
    参数:
        yyxx: 边界框坐标 (y1, y2, x1, x2)
        H: 原始图像高度
        W: 原始图像宽度
        h: 目标图像高度
        w: 目标图像宽度
    
    返回:
        调整后的边界框坐标 (y1, y2, x1, x2)
    """
    y1, y2, x1, x2 = yyxx
    y1 = int(y1 * h / H)
    y2 = int(y2 * h / H)
    x1 = int(x1 * w / W)
    x2 = int(x2 * w / W)
    y1, y2 = min(y1, h), min(y2, h)
    x1, x2 = min(x1, w), min(x2, w)
    return y1, y2, x1, x2

def get_bbox_from_mask(mask):
    """
    获取掩码的边界框。
    
    参数:
        mask: 二值掩码（numpy数组）
    
    返回:
        边界框坐标 (y1, y2, x1, x2)
    """
    h, w = mask.shape[0], mask.shape[1]
    rows = np.any(mask, axis = 1)
    cols = np.any(mask, axis = 0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (y1, y2, x1, x2)

def expand_bbox(mask, yyxx, ratio=[1.2, 2.0], min_crop=0):
    """
    对边界框进行扩展（放大），确保不超出图像范围，并满足最小尺寸要求。
    
    参数:
        mask: 二值掩码（用于获取图像尺寸，实际未直接使用掩码内容）
        yyxx: 原始边界框坐标 (y1, y2, x1, x2)
        ratio: 扩展比例范围 [min_ratio, max_ratio]（会被随机选择）
        min_crop: 扩展后的边界框的最小宽度和高度（像素）
    
    返回:
        扩展后的边界框坐标 (y1, y2, x1, x2)
    """
    y1, y2, x1, x2 = yyxx
    h, w = mask.shape[0], mask.shape[1]
    ratio = np.random.randint(ratio[0] * 10, ratio[1] * 10) / 10
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    h_expand = h * ratio
    w_expand = w * ratio
    h_expand = max(h_expand, min_crop)
    w_expand = max(w_expand, min_crop)

    x1 = max(0, int(xc - w_expand / 2))
    x2 = min(w, int(xc + w_expand / 2))
    y1 = max(0, int(yc - h_expand / 2))
    y2 = min(h, int(yc + h_expand / 2))
    return (y1, y2, x1, x2)

def box2squre(image, box):
    """
    将边界框转换为正方形。
    
    参数:
        image: 输入图像（numpy数组）
        box: 边界框坐标 (y1, y2, x1, x2)
    
    返回:
        正方形边界框坐标 (y1, y2, x1, x2)
    """
    y1, y2, x1, x2 = box
    H, W = image.shape[0], image.shape[1]
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    max_side = max(w, h)
    x1 = max(0, xc - max_side // 2)
    x2 = min(W, xc + max_side // 2)
    y1 = max(0, yc - max_side // 2)
    y2 = min(H, yc + max_side // 2)
    return (y1, y2, x1, x2)

def pad_to_square(image, pad_value = 255, random = False):
    """
    将图像填充为正方形。
    
    参数:
        image: 输入图像（numpy数组）
        pad_value: 填充值（默认为255）
        random: 是否随机填充
    
    返回:
        填充后的图像
    """
    h, w = image.shape[0], image.shape[1]
    if h == w:
        return image
    padd = abs(h - w)
    if random:
        padd_1 = int(np.random.randint(0, padd))
    else:
        padd_1 = padd // 2
    padd_2 = padd - padd_1
    if h > w:
        pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
    else:
        pad_param = ((padd_1, padd_2), (0, 0), (0, 0))
    image = np.pad(image, pad_param, mode = 'constant', constant_values = pad_value)
    return image

def box_in_box(small_box, big_box):
    """
    将小边界框的坐标转换为相对于大边界框的局部坐标。
    
    参数:
        small_box: 小边界框坐标 (y1, y2, x1, x2)
        big_box: 大边界框坐标 (y1_, y2_, x1_, x2_)
    
    返回:
        小边界框相对于大边界框的局部坐标 (y1_rel, y2_rel, x1_rel, x2_rel)
    """
    y1, y2, x1, x2 = small_box
    y1_, _, x1_, _ = big_box
    y1_rel, y2_rel, x1_rel, x2_rel = y1 - y1_, y2 - y1_, x1 - x1_, x2 - x1_
    return (y1_rel, y2_rel, x1_rel, x2_rel)

def shuffle_image(image, N):
    """
    将图像分成N*N个块，然后随机打乱这些块的顺序，最后重新组合成一个图像。
    
    参数:
        image: 输入图像（numpy数组）
        N: 分块数
    
    返回:
        打乱顺序后的图像
    """
    height, width = image.shape[:2]

    block_height = height // N
    block_width = width // N
    blocks = []
    for i in range(N):
        for j in range(N):
            block = image[i * block_height : (i + 1) * block_height, 
                          j * block_width : (j + 1) * block_width]
            blocks.append(block)
    
    np.random.shuffle(blocks)
    shuffled_image = np.zeros((height, width, 3), dtype = np.uint8)

    for i in range(N):
        for j in range(N):
            shuffled_image[i * block_height : (i + 1) * block_height, 
                          j * block_width : (j + 1) * block_width] = blocks[i * N + j]
    
    return shuffled_image

def get_mosaic_mask(image, fg_mask, N = 16, ratio = 0.5):
    """
    对图像进行马赛克处理，随机选择部分图像块进行模糊化，并保留前景掩码区域。
    
    参数:
        image: 输入图像（NumPy 数组，形状为 (height, width, channels)）
        fg_mask: 前景掩码（二值掩码，值为 0 或 1，表示前景和背景）
        N: 将图像划分为 N×N 的块
        ratio: 马赛克处理的块比例（0~1，表示需要马赛克的块数占总块数的比例）
    
    返回:
        noise_mask: 处理后的图像（马赛克 + 噪声）
    """
    ids = [i for i in range(N * N)]
    masked_number = int(N * N * ratio)
    masked_id = np.random.choice(ids, masked_number, replace=False)

    height, width = image.shape[:2]
    mask = np.ones((height, width))

    block_height = height // N
    block_width = width // N
    block_id = 0
    for i in range(N):
        for j in range(N):
            if block_id in masked_id:
                mask[i * block_height : (i + 1) * block_height, 
                          j * block_width : (j + 1) * block_width] = 0
            block_id += 1
    mask *= fg_mask
    mask3 = np.stack([mask, mask, mask], -1).copy().astype(np.uint8)
    noise = q_x(image)
    noise_mask = image * mask3 + noise * (1 - mask3)
    return noise_mask

