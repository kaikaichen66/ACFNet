# -*- coding: utf-8 -*-
from ultralytics import YOLO
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 核心修改：同时加载结构配置文件和预训练权重文件
    # model() 里的第一个参数是权重路径，第二个是配置路径
    model = YOLO(
        model=r'/home/chenkaikai/PythonProjects/YOLOv8v10v11_Demov10/ultralytics/cfg/models/addv11/ACF_SCE_BSU2.yaml'
    ).load(r'/home/chenkaikai/PythonProjects/YOLOv8v10v11_Demov10/yolo11n.pt') 
    
    # 或者也可以直接写成：
    # model = YOLO(r'/home/chenkaikai/PythonProjects/YOLOv8v10v11_Demov10/yolov11n.pt')
    # 但因为你有自定义的 .yaml（addv11），所以建议用 .load() 的方式确保结构准确
    
    # 开始训练
    model.train(
        data='/home/chenkaikai/PythonProjects/YOLOv8v10v11_Demov10/ultralytics/cfg/datasets/PV-Multi-Defect.yaml',
        #data='/home/chenkaikai/PythonProjects/YOLOv8v10v11_Demov10/datasets/NEU-DET/NEU-DET.yaml',
        cache=False,      # 是否缓存数据集
        imgsz=640,        # 图像尺寸
        epochs=300,       # 训练轮数
        batch=16,         # 批次大小
        close_mosaic=10,  # 建议最后 10 轮关闭 Mosaic，有利于精度收敛
        workers=8,        # 数据加载线程数
        patience=50,      # 早停机制
        device='1',       # 指定 GPU 设备 0
        optimizer='SGD',  # 优化器
        project='train_yolov11', # 建议指定项目名称，方便找结果
        name='exp_neu_det'       # 实验名称
    )