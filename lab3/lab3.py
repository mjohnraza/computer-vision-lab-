import cv2
import numpy as np

# --- Task 1: Scale (Linear) ---
def task1(img):
    h, w = img.shape[:2]
    s = 3.0
    S = np.diag([s, s])
    t = np.float32([[(w / 2) * (1 - s)], [(h / 2) * (1 - s)]])
    M = np.hstack([S, t])
    return cv2.warpAffine(img, M, (w, h))

# --- Task 2: Rotation (Linear) ---
def task2(img):
    h, w = img.shape[:2]
    th = np.radians(-45)
    c, s = abs(np.cos(th)), abs(np.sin(th))
    nw, nh = int(w * c + h * s), int(h * c + w * s)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -45, 1.0)
    M[:, 2] += [(nw - w) / 2, (nh - h) / 2]
    return cv2.warpAffine(img, M, (nw, nh))

# --- Task 3: Shear (Linear) ---
def task3(img):
    h, w = img.shape[:2]
    sh = 0.35
    S = np.float32([[1, sh], [0, 1]])
    t = np.float32([[-sh * w / 2], [0]])
    M = np.hstack([S, t])
    return cv2.warpAffine(img, M, (w, h))

# --- Task 4: Translation (Affine) ---
def task4(img):
    h, w = img.shape[:2]
    T = np.float32([[1, 0, 150], [0, 1, 80], [0, 0, 1]])
    return cv2.warpAffine(img, T[:2], (w, h))

# --- Task 5: Rigid (Euclidean) ---
def task5(img):
    h, w = img.shape[:2]
    th = np.radians(30)
    c, s = np.cos(th), np.sin(th)
    R = np.float32([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.float32([[1, 0, 40], [0, 1, 80], [0, 0, 1]])
    M = (T @ R)[:2]
    return cv2.warpAffine(img, M, (w, h))

# --- Task 6: Similarity ---
def task6(img):
    h, w = img.shape[:2]
    s = 1.8
    th = np.radians(-35)
    c, sn = np.cos(th), np.sin(th)
    S = np.diag([s, s, 1.0])
    R = np.float32([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    T = np.float32([[1, 0, 100], [0, 1, 90], [0, 0, 1]])
    M = (T @ R @ S)[:2]
    return cv2.warpAffine(img, M, (w, h))

# --- Task 7: General Affine ---
def task7(img):
    h, w = img.shape[:2]
    p1 = np.float32([[50, 50], [200, 50], [100, 200]])
    p2 = np.float32([[70, 100], [220, 40], [150, 240]])
    A = np.c_[p1, np.ones(3)]
    u = np.linalg.solve(A, p2[:, 0])
    v = np.linalg.solve(A, p2[:, 1])
    M = np.float32([u, v])
    return cv2.warpAffine(img, M, (w, h))

# --- Task 8: Perspective ---
def task8(img):
    h, w = img.shape[:2]
    src = np.float32([[120, 80], [240, 80], [310, 310], [50, 310]])
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w, h))

# --- Task 9: Homography Panorama ---
def task9(cam1, cam2):
    h, w = cam1.shape[:2]
    p2 = np.float32([[60, 60], [160, 60], [160, 280], [60, 280]])
    p1 = np.float32([[200, 70], [300, 50], [310, 290], [210, 270]])
    H, _ = cv2.findHomography(p2, p1)
    pano = cv2.warpPerspective(cam2, H, (w * 2, h))
    pano[:, :w] = np.where(cam1 > 0, cam1, pano[:, :w])
    return pano

# --- Task 10: Full Hierarchy ---
def task10(wall, paint):
    h, w = paint.shape[:2]
    th = np.radians(15)
    c, s = np.cos(th), np.sin(th)
    S = np.diag([0.5, 0.5, 1.0])
    R = np.float32([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.float32([[1, 0, 80], [0, 1, 60], [0, 0, 1]])
    step12 = cv2.warpAffine(paint, (T @ R @ S)[:2], (w, h))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[100, 60], [280, 90], [250, 310], [80, 270]])
    H = cv2.getPerspectiveTransform(src, dst)
    res = cv2.warpPerspective(step12, H, (w, h))
    wall[res > 0] = res[res > 0]
    return wall

if __name__ == "__main__":
    # Test images
    d = 360
    blank = np.zeros((d, d, 3), dtype=np.uint8)
    
    cv2.circle(blank, (d // 2, d // 2), 60, (0, 255, 180), 2)
    t1 = task1(blank)
    
    cv2.rectangle(blank, (100, 100), (260, 260), (255, 160, 50), -1)
    t2 = task2(blank)
    
    t3 = task3(blank)
    t4 = task4(blank)
    t5 = task5(blank)
    t6 = task6(blank)
    t7 = task7(blank)
    t8 = task8(blank)
    t9 = task9(blank, blank)
    t10 = task10(blank.copy(), blank)
    
    print("All tasks 1-10 complete.")
