# Blur Detection & Visual Quality Assessment Module

本模块是 `AVTSE_Dataset_Builder` 数据集构建工具链的核心质量控制（Quality Control）环节。

视听觉目标说话人抽取（Audio-Visual Target Speaker Extraction）任务对人脸及唇部区域的视觉清晰度要求极高。模糊、失真或低分辨率的人脸帧会导致唇语对齐与特征提取失效。本模块提供两种互补的清晰度评估算法供自动筛选使用。