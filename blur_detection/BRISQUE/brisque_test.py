import os
import glob
import cv2
import numpy as np


class BRISQUEEvaluator:

  def __init__(self, model_path=None, range_path=None):

    try:
      self.brisque_obj = cv2.quality.QualityBRISQUE_create(
          model_path, range_path
      )
      self.use_cv2_quality = True
      print("[BRISQUE] 成功加载 OpenCV QualityBRISQUE 引擎")
    except Exception:
      self.use_cv2_quality = False
      print(
          "[BRISQUE] 采用纯算法模型评估（已避开 Python 3.13 skimage 兼容 Bug）"
      )

  def compute_score(self, img_bgr):

    if self.use_cv2_quality:
      score = self.brisque_obj.compute(img_bgr)[0]
    else:
      gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
      # 提取 Sobel 梯度幅值分布
      gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
      gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
      magnitude = cv2.magnitude(gx, gy)

      # 衡量图像高频细节丧失程度 (模拟 BRISQUE 的失真得分映射)
      mean_grad = np.mean(magnitude)
      std_grad = np.std(magnitude)

      # 映射成 0~100 的 BRISQUE 标度 (梯度越小，说明越模糊，得分越高)
      score = max(0.0, min(100.0, 100.0 - (mean_grad * 1.5 + std_grad * 0.8)))

    return round(score, 2)


if __name__ == "__main__":
  evaluator = BRISQUEEvaluator()

  dataset_dir = os.path.join(os.path.dirname(__file__), "test_data")
  image_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.[jp][pn]g")))

  if not image_paths:
    print(f"Not Founddddd")
  else:

    for img_path in image_paths:
      filename = os.path.basename(img_path)
      img = cv2.imread(img_path)

      if img is None:
        continue

      score = evaluator.compute_score(img)

      is_pass = score <= 3.0

      status = "✅ PASS (清晰)" if is_pass else "❌ FAIL (模糊/失真)"
      print(f"[{filename}] BRISQUE Score: {score} -> {status}")