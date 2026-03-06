from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2


def compute_metrics(gt, pred):

    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

    psnr = peak_signal_noise_ratio(gt, pred)

    ssim = structural_similarity(
        gt_gray,
        pred_gray
    )

    return psnr, ssim