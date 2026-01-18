import cv2
import os

def save_rois(image, roi_list, output_dir="./roi_region"):
    """
    遍历 ROI 列表并保存裁剪区域
    :param image: 原始图片数组 (形状为 2048, 1676, 3)
    :param roi_list: 包含 bbox 等信息的列表
    :param output_dir: 保存目录
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, roi in enumerate(roi_list):
        # 1. 解析 bbox [x_min, y_min, x_max, y_max]
        # 根据你提供的数据 [145, 445, 885, 965]
        x_min, y_min, x_max, y_max = map(int, roi["bbox"])

        # 2. 裁剪图片
        # 注意：NumPy 数组切片顺序是 [y_start:y_end, x_start:x_end]
        roi_img = image[y_min:y_max, x_min:x_max]

        # 3. 检查裁剪结果是否有效（防止越界导致空图片）
        if roi_img.size == 0:
            print(f"跳过 ROI {i}: 裁剪区域为空，请检查坐标。")
            continue

        # 4. 生成文件名并保存
        # 可以根据类别和索引命名，例如: tumor_region_0.png
        type = roi["class"]
        file_name = f"{roi['class']}_{i}_{type}.png"
        save_path = os.path.join(output_dir, file_name)

        # 保存为 PNG
        cv2.imwrite(save_path, roi_img)
        print(f"已保存: {save_path}, 尺寸: {roi_img.shape[:2]}")