import os
import cv2
from load_data import LoLIDataset
from enhancement import enhance_clahe
from metrics import compute_metrics

LOW_PATH = "dataset/low"
GT_PATH = "dataset/normal"

RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

dataset = LoLIDataset(LOW_PATH, GT_PATH)

total_psnr = 0
total_ssim = 0

print("Running enhancement experiment...\n")

for i in range(len(dataset)):

    low_img, gt_img, name = dataset.get_pair(i)

    enhanced = enhance_clahe(low_img)

    psnr, ssim = compute_metrics(gt_img, enhanced)

    total_psnr += psnr
    total_ssim += ssim

    comparison = cv2.hconcat([
        low_img,
        enhanced,
        gt_img
    ])

    save_path = os.path.join(RESULT_DIR, name)
    cv2.imwrite(save_path, comparison)

    print(f"{name} | PSNR: {psnr:.2f} | SSIM: {ssim:.3f}")

avg_psnr = total_psnr / len(dataset)
avg_ssim = total_ssim / len(dataset)

print("\n======================")
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.3f}")
print("======================")