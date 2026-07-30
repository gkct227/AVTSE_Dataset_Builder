import os
import cv2
import numpy as np


def check_frame_quality(
    image, laplacian_thresh=80.0, low_bright=30, high_bright=225
):
    if image is None or image.size == 0:
        return {"pass": False, "reason": "Invalid/Empty Image"}

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian_map = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian_map.var()
    is_blur = blur_score < laplacian_thresh

    mean_bright = float(np.mean(gray))
    is_bad_lighting = (mean_bright < low_bright) or (
        mean_bright > high_bright
    )

    is_pass = (not is_blur) and (not is_bad_lighting)

    reason = "PASS"
    if is_blur and is_bad_lighting:
        reason = "Blur & Bad Lighting"
    elif is_blur:
        reason = "Blurry"
    elif is_bad_lighting:
        reason = "Bad Lighting"

    return {
        "pass": is_pass,
        "reason": reason,
        "blur_score": round(blur_score, 2),
        "bright_score": round(mean_bright, 2),
        "laplacian_map": laplacian_map,
    }

if __name__ == "__main__":

    clean_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(
        clean_img, (30, 30), (270, 270), (255, 255, 255), -1
    )
    cv2.circle(clean_img, (150, 150), 80, (0, 0, 0), -1)  # 黑色圆形
    cv2.putText(
        clean_img,
        "HD Text",
        (80, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )

    blurry_img = cv2.GaussianBlur(clean_img, (31, 31), 0)

    clean_res = check_frame_quality(clean_img)
    blurry_res = check_frame_quality(blurry_img)

    os.makedirs("output_vis", exist_ok=True)
    cv2.imwrite("output_vis/01_clean_input.png", clean_img)
    cv2.imwrite("output_vis/02_blurry_input.png", blurry_img)

    clean_lap_vis = np.uint8(np.absolute(clean_res["laplacian_map"]))
    blurry_lap_vis = np.uint8(np.absolute(blurry_res["laplacian_map"]))

    cv2.putText(
        clean_img,
        f"Score: {clean_res['blur_score']} (PASS)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        blurry_img,
        f"Score: {blurry_res['blur_score']} (FAIL)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    cv2.imwrite("output_vis/01_clean_with_score.png", clean_img)
    cv2.imwrite("output_vis/02_blurry_with_score.png", blurry_img)
    cv2.imwrite("output_vis/03_clean_edge_response.png", clean_lap_vis)
    cv2.imwrite("output_vis/04_blurry_edge_response.png", blurry_lap_vis)

    print(f"\n[清晰图] 结果: {clean_res['reason']} | 得分: {clean_res['blur_score']}")
    print(
        f"[模糊图] 结果: {blurry_res['reason']} | 得分: {blurry_res['blur_score']}"
    )
    print("\n 可视化图片已成功保存到当前目录下的 `output_vis/` 文件夹中！")