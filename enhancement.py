import cv2

def enhance_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_enhanced = clahe.apply(l)

    merged = cv2.merge((l_enhanced, a, b))
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return enhanced_img