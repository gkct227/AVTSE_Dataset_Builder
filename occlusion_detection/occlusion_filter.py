import os
import sys
import glob
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

repo_path = os.path.join(os.path.dirname(__file__), "face_parsing_repo")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from model import BiSeNet


class BiSeNetOcclusionFilter:
    COLOR_MAP = np.array([
        [0, 0, 0],  # 0: background
        [204, 0, 0],  # 1: skin
        [0, 255, 255],  # 2: l_brow
        [255, 204, 204],  # 3: r_brow
        [51, 51, 255],  # 4: l_eye
        [204, 0, 204],  # 5: r_eye
        [204, 204, 0],  # 6: eye_g
        [102, 51, 0],  # 7: l_ear
        [255, 0, 0],  # 8: r_ear
        [0, 204, 204],  # 9: ear_r
        [76, 153, 0],  # 10: nose
        [102, 204, 0],  # 11: mouth
        [255, 255, 0],  # 12: u_lip
        [0, 0, 153],  # 13: l_lip
        [255, 153, 51],  # 14: neck
        [0, 51, 0],  # 15: neck_l
        [0, 204, 0],  # 16: cloth
        [0, 0, 204],  # 17: hair
        [255, 51, 153]  # 18: hat
    ], dtype=np.uint8)

    def __init__(self, weight_path=None, occlusion_thresh=0.20, device="cpu"):
        self.occlusion_thresh = occlusion_thresh
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.net = BiSeNet(n_classes=19).to(self.device)

        if weight_path is None:
            weight_path = os.path.join(repo_path, "res/cp/79999_iter.pth")

        print(f"[BiSeNet]: {weight_path}")
        self.net.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def predict_parsing_map(self, img_bgr):
        """使用官方推理逻辑"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        image = pil_img.resize((512, 512), Image.BILINEAR)
        img_tensor = self.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.net(img_tensor)[0]
            parsing_map = out.squeeze(0).cpu().numpy().argmax(axis=0)

        h, w = img_bgr.shape[:2]
        if (h, w) != (512, 512):
            parsing_map = cv2.resize(parsing_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        return parsing_map

    def evaluate_occlusion(self, parsing_map):
        # 1. 查找分割出来的嘴唇像素 (11: mouth interior, 12: upper lip, 13: lower lip)
        lip_pixels = np.where(np.isin(parsing_map, [11, 12, 13]))

        # 如果完全找不到嘴唇像素：
        if len(lip_pixels[0]) < 20:  # 嘴唇像素太少或没有
            face_pixels = np.where(np.isin(parsing_map, [1, 2, 3, 4, 5, 10]))
            if len(face_pixels[0]) > 0:
                y_max = np.max(face_pixels[0])
                x_m = np.mean(face_pixels[1])
                h, w = parsing_map.shape
                roi_box = (
                    max(0, int(x_m - w * 0.15)),
                    max(0, int(y_max * 0.65)),
                    min(w, int(x_m + w * 0.15)),
                    min(h, int(y_max * 0.85)),
                )
            else:
                h, w = parsing_map.shape
                roi_box = (
                    int(w * 0.35),
                    int(h * 0.65),
                    int(w * 0.65),
                    int(h * 0.85),
                )

            # 嘴唇没有
            return {
                "pass": False,
                "occlusion_rate": 1.0,  # 100% 被遮挡
                "roi_box": roi_box,
            }

        # 2. 如果找到了嘴唇，以嘴唇几何中心为基准
        y_m, x_m = np.mean(lip_pixels[0]), np.mean(lip_pixels[1])
        h, w = parsing_map.shape

        # 只取嘴唇周围很小的范围
        box_h, box_w = int(h * 0.12), int(w * 0.18)
        y_start = max(0, int(y_m - box_h * 0.5))
        y_end = min(h, int(y_m + box_h * 0.5))
        x_start = max(0, int(x_m - box_w * 0.5))
        x_end = min(w, int(x_m + box_w * 0.5))

        roi_mask = parsing_map[y_start:y_end, x_start:x_end]

        valid_classes = [1, 11, 12, 13]
        total_pixels = roi_mask.size
        valid_pixels = np.isin(roi_mask, valid_classes).sum()

        occlusion_rate = float(total_pixels - valid_pixels) / (
                total_pixels + 1e-6
        )
        is_pass = occlusion_rate <= self.occlusion_thresh

        return {
            "pass": is_pass,
            "occlusion_rate": round(occlusion_rate, 4),
            "roi_box": (x_start, y_start, x_end, y_end),
        }

    def render_overlay_vis(self, orig_bgr, parsing_map):
        """渲染Mask 图像"""
        color_vis = self.COLOR_MAP[parsing_map]
        color_vis_bgr = cv2.cvtColor(color_vis, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(orig_bgr, 0.5, color_vis_bgr, 0.5, 0)


if __name__ == "__main__":
    detector = BiSeNetOcclusionFilter(occlusion_thresh=0.20)

    output_dir = os.path.join(os.path.dirname(__file__), "output_vis")
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir = os.path.join(os.path.dirname(__file__), "test_data")
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.[jp][pn]g")))

    print("\n--- 运行 BiSeNet 下半脸/唇部遮挡评估并保存结果图 ---")
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1. Parsing Mask
        parsing_map = detector.predict_parsing_map(img)

        # 2. 遮挡率
        res = detector.evaluate_occlusion(parsing_map)

        # 3. 效果图
        overlay = detector.render_overlay_vis(img, parsing_map)

        # 4. ROI 框和标记
        x1, y1, x2, y2 = res["roi_box"]
        color = (0, 255, 0) if res["pass"] else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"PASS: {res['pass']} | Occ: {res['occlusion_rate'] * 100:.1f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        out_path = os.path.join(output_dir, f"result_{idx + 1}_{filename}")
        cv2.imwrite(out_path, overlay)

        print(f"[{filename}] PASS={res['pass']} | 遮挡率={res['occlusion_rate'] * 100:.1f}% -> 已保存至 {out_path}")