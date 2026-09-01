# -*- coding: utf-8 -*-
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO('configs/ACFNet.yaml')

    model.train(
        data='datasets/PV-Multi-Defect.yaml',
        cache=False,
        imgsz=640,
        epochs=300,
        batch=16,
        close_mosaic=10,
        workers=8,
        patience=50,
        device='1',
        optimizer='SGD',
        seed=0,
        deterministic=True,
        project='train_PV',
        name='exp_PV_Multi_Defect_seed0'
    )
