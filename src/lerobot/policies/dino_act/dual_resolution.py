import torch
import torch.nn.functional as F
import cv2
def crop_gripper_area(images, crop_h=280, crop_w=420, center_y=320, center_x=320):
    """
    参数说明:
    images: 输入张量 [B, C, H, W]，此时为 [B, 3, 480, 640]
    crop_h: 你想要的裁剪高度 (大小参数 1)
    crop_w: 你想要的裁剪宽度 (大小参数 2)
    center_y: 裁剪区域中心的纵坐标 (位置参数 1, 范围 0-480)
    center_x: 裁剪区域中心的横坐标 (位置参数 2, 范围 0-640)
    """
    # 1. 计算裁剪边界
    h, w = image.shape[:2] # 获取当前图片的实际高宽
    
    # 计算边界
    y1 = max(0, center_y - crop_h // 2)
    y2 = min(h, y1 + crop_h)
    x1 = max(0, center_x - crop_w // 2)
    x2 = min(w, x1 + crop_w)
    
    # 再次检查尺寸，防止在边缘处切出的尺寸不对
    if y2 == h: y1 = max(0, h - crop_h)
    if x2 == w: x1 = max(0, w - crop_w)

    # OpenCV 的切片方式: [H, W, C]
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image




if __name__ == "__main__":
    # 模拟一个 Batch 的图片 (Batch=4, Channel=3, Height=480, Width=640)
    image = cv2.imread("outputs/captured_images/opencv_2.png")
    if image is None:
        print("图片读取失败，请检查路径！")
    else:
        # 2. 裁剪
        cropped = crop_gripper_area(image)
        
        # 3. 显示
        cv2.imshow("Original", image)
        cv2.imshow("Cropped", cropped)
        print(f"Original shape: {image.shape}") # (480, 640, 3)
        print(f"Cropped shape: {cropped.shape}")   # (300, 400, 3)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# img_cropped = crop_gripper_area(images, my_crop_h, my_crop_w, my_center_y, my_center_x)