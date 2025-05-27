import numpy as np
import cv2

rng = np.random.default_rng(42)

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

def get_bbox_from_mask(mask):
    """
    获取掩码的边界框。
    
    参数:
        mask: 二值掩码（numpy数组）
    
    返回:
        边界框坐标 (y1, y2, x1, x2)
    """
    h, w = mask.shape[0], mask.shape[1]
    if mask.sum() < 10:
        return 0, h, 0, w
    rows = np.any(mask, axis = 1)
    cols = np.any(mask, axis = 0)
    y1, y2 = np.nonzero(rows)[0][[0, -1]]
    x1, x2 = np.nonzero(cols)[0][[0, -1]]
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
    ratio = rng.integers(ratio[0] * 10, ratio[1] * 10) / 10
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
        padd_1 = int(rng.integers(0, padd))
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

def get_random_structure(size):
    """
    根据给定的大小生成随机形状的结构元素。

    参数:
        size: 结构元素的大小

    返回:
        结构元素
    """
    choice = rng.integers(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size // 2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 2, size))

def random_dilate(seg, min = 3, max = 10):
    """
    随机膨胀函数

    参数:
        seg: 输入图像（NumPy 数组，形状为 (height, width, channels)）
        min: 结构元素的最小大小（默认为 3）
        max: 结构元素的最大大小（默认为 10）

    返回:
        seg: 处理后的图像
    """
    size = rng.integers(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg, kernel, iterations = 1)
    return seg

def random_erode(seg, min = 3, max = 10):
    """
    随机腐蚀函数

    参数:
        seg: 输入图像（NumPy 数组，形状为 (height, width, channels)）
        min: 结构元素的最小大小（默认为 3）
        max: 结构元素的最大大小（默认为 10）

    返回:
        seg: 处理后的图像
    """
    size = rng.integers(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg, kernel, iterations = 1)
    return seg

def compute_iou(seg, gt):
    """
    计算交并比

    参数:
        seg: 分割结果（二值掩码）
        gt: 真实标签（二值掩码）

    返回:
        iou: 交并比
    """
    intersection = seg * gt
    union = seg + gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def select_max_region(mask):
    """
    选择最大连通区域

    参数:
        mask: 输入图像（二值掩码）

    返回:
        max_region: 最大连通区域
    """
    nums, labels, stats, _ = cv2.connectedComponetsWithStats(mask, connectivity = 8)
    max_area = 0
    max_label = 0
    for i in range(1, nums):
        if stats[i, 4] > max_area:
            max_area = stats[i, 4]
            max_label = i

    max_region = np.where(labels == max_label, 1, 0)
    return max_region.astype(np.uint8)

def perturb_mask(gt, min_iou = 0.3, max_iou = 0.99):
    """
    对真实标签进行随机扰动，直到达到目标交并比

    参数:
        gt: 真实标签（二值掩码）
        min_iou: 目标交并比的最小值（默认为 0.3）
        max_iou: 目标交并比的最大值（默认为 0.99）

    返回:
        seg: 扰动后的分割结果
    """
    iou_target = rng.uniform(min_iou, max_iou)
    h, w = gt.shape
    gt = gt.astype(np.uint8)
    seg = gt.copy()

    if h <= 2 or w <= 2:
        print("GT too small, return ing original")
        return seg
    
    for _ in range(250):
        for _ in range(4):
            lx, ly = rng.integers(w), rng.integers(h)
            rx, ry = rng.integers(lx + 1, w + 1), rng.integers(ly + 1, h + 1)

            if rng.random() < 0.1:
                cx = (lx + rx) // 2
                cy = (ly + ry) // 2
                seg[cy, cx] = rng.integers(2) * 255

            if rng.random() < 0.5:
                seg[ly : ry, lx : rx] = random_dilate(seg[ly : ry, lx : rx])
            else:
                seg[ly : ry, lx : rx] = random_erode(seg[ly : ry, lx : rx])
            
            seg = np.logical_or(seg, gt).astype(np.uint8)

        if compute_iou(seg, gt) < iou_target:
            break

    seg = select_max_region(seg.astype(np.uint8))
    return seg.astype(np.uint8)