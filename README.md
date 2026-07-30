# AVTSE Dataset Builder (Audio-Visual Target Speaker Extraction)

`AVTSE_Dataset_Builder` 是一个专为**视听觉目标说话人抽取**任务设计的高质量数据集构建与自动化筛选工具链。

---

## 📁 项目目录结构

* **`blurdetection/`**: 人脸清晰度/模糊度检测与筛选模块。
* **`occlusion_detection/`**: 基于 BiSeNet (19-class Face Parsing) 的下半脸与唇部遮挡自动化过滤模块。

---

## 🚀 快速开始

### 1. 模糊度检测 (Blur Detection)
详见 [`blurdetection/README.md`](blurdetection/README.md)。

### 2. 唇部遮挡检测 (Occlusion Filter)
```bash
cd occlusion_detection
python occlusion_filter.py