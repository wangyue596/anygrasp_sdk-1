import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 用于调整相机的旋转
def rotation_matrix_xyz(theta_x, theta_y, theta_z):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

def apply_rotation(pcd, rotation_matrix):
    # 转换点云中的每个点
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) @ rotation_matrix.T)
    return pcd

# 用于调整相机的平移
def apply_translation(pcd, translation_vector):
    # 转换点云中的每个点
    translated_points = np.asarray(pcd.points) + translation_vector
    pcd.points = o3d.utility.Vector3dVector(translated_points)
    return pcd

def point_cloud_to_images(pcd):
    # 从点云中获取点
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # 确保点云中包含颜色数据

    # 定义相机参数
    width, height = 640, 480
    fx, fy = 525.0, 525.0
    cx, cy = width / 2, height / 2

    # 创建图像
    # depth_image = np.ones((height, width), dtype=np.float32)
    # rgb_image = np.ones((height, width, 3), dtype=np.uint8)
    depth_image = np.zeros((height, width), dtype=np.float32)
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 将点云投影到图像平面
    for point, color in zip(points, colors):
        x, y, z = point
        if z > 0.001:  # 避免除以接近零的值
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)
        
            if 0 <= u < width and 0 <= v < height:
                depth_image[v, u] = z
                # 颜色赋值时交换红色和蓝色通道
                rgb_image[v, u] = [(color[2] * 255), (color[1] * 255), (color[0] * 255)]

    # 使用OpenCV保存图像
    cv2.imwrite('rgb_image.png', rgb_image)
    # cv2.imwrite('depth_image.png', depth_image.astype(np.uint16))  # 归一化以保存
    cv2.imwrite('depth_image.png', depth_image / depth_image.max() * 255)  # 归一化以保存
    
    # 使用matplotlib显示图像
    plt.figure(figsize=(12, 6))

    # 显示 RGB 图像
    rgb_image_rgb = rgb_image[:, :, [2, 1, 0]]  # 重新排列BGR为RGB
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image_rgb)
    plt.title('RGB Image')
    plt.axis('off')  # 关闭坐标轴

    # 显示深度图像
    plt.subplot(1, 2, 2)
    # 归一化深度图像并转换为整数以便显示
    depth_normalized = (depth_image / depth_image.max() * 255).astype(np.uint8)
    plt.imshow(depth_normalized, cmap='gray')  # 使用灰度色图显示
    plt.title('Depth Image')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    # 使用示例
    pcd = o3d.io.read_point_cloud("nontextured_simplified(1).ply")

    # 定义旋转角度
    theta_x, theta_y, theta_z = np.radians(30), np.radians(30), np.radians(60)  # 示例角度
    # theta_x, theta_y, theta_z = np.radians(60), np.radians(0), np.radians(120)  # glass角度
    # theta_x, theta_y, theta_z = np.radians(60), np.radians(0), np.radians(70)  # 1角度
    # theta_x, theta_y, theta_z = np.radians(60), np.radians(0), np.radians(50)  # 2角度
    # 获取旋转矩阵
    rotation_mat = rotation_matrix_xyz(theta_x, theta_y, theta_z)
    # 应用旋转
    pcd_rotated = apply_rotation(pcd, rotation_mat)

    # 调整相机的平移
    # x_offset = -0.55  # glass
    # x_offset = -0.55   # 1
    # y_offset = 0.5    # 1
    # z_offset = 0.55   # 2
    # x_offset = -1.55  # 2
    # y_offset = -0.55  # 2
    x_offset = 0  # 初始
    y_offset = 0  # 初始
    z_offset = 0.55   # 初始

    translation_vector = np.array([x_offset, y_offset, z_offset])
    pcd_translated = apply_translation(pcd_rotated, translation_vector)

    point_cloud_to_images(pcd_translated)