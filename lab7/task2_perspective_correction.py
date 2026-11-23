"""
Task 2: Perspective Correction (Unwarping) / Inverse Perspective Mapping
=========================================================================
Author: Muhammad John Raza
Roll No: 23k-0069
FAST NUCES Karachi - Department of Artificial Intelligence
Course: Computer Vision Lab (Lab 07)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_path(rel_path):
    if os.path.exists(rel_path):
        return rel_path
    if rel_path.startswith("lab7/") and os.path.exists(rel_path[5:]):
        return rel_path[5:]
    if os.path.exists(os.path.join("lab7", rel_path)):
        return os.path.join("lab7", rel_path)
    if os.path.basename(os.getcwd()) == "lab7" and rel_path.startswith("lab7/"):
        return rel_path[5:]
    return rel_path

output_dir = "outputs" if os.path.basename(os.getcwd()) == "lab7" else "lab7/outputs"
os.makedirs(output_dir, exist_ok=True)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def apply_perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    maxWidth = max(maxWidth, 100)
    maxHeight = max(maxHeight, 100)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped, M, (maxWidth, maxHeight)

def auto_detect_document_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 10000:
            return approx.reshape(4, 2).astype("float32")
    return np.float32([[260, 90], [820, 160], [710, 720], [130, 580]])

def forward_projection_demo(fronto_img, desk_img, target_quad):
    h_f, w_f = fronto_img.shape[:2]
    src_rect = np.float32([[0, 0], [w_f - 1, 0], [w_f - 1, h_f - 1], [0, h_f - 1]])
    dst_quad = order_points(target_quad)
    H_fwd = cv2.getPerspectiveTransform(src_rect, dst_quad)
    h_d, w_d = desk_img.shape[:2]
    projected = cv2.warpPerspective(fronto_img, H_fwd, (w_d, h_d))
    mask = np.zeros((h_d, w_d), dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(dst_quad)], 255)
    mask_3d = cv2.merge([mask, mask, mask]) / 255.0
    blended = (projected * mask_3d + desk_img * (1.0 - mask_3d)).astype(np.uint8)
    return blended, H_fwd

def run_perspective_correction():
    print("=================================================================")
    print("      LAB 07 - TASK 2: PERSPECTIVE CORRECTION (UNWARPING)       ")
    print("      Author: Muhammad John Raza (23k-0069)                     ")
    print("=================================================================")

    img_path = get_path("lab7/assets/book_slanted.jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found at {img_path}")

    corners = auto_detect_document_corners(img)
    print(f"[*] Detected / Ordered Corners:")
    ordered_corners = order_points(corners)
    labels = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Right (BR)", "Bottom-Left (BL)"]
    for lbl, pt in zip(labels, ordered_corners):
        print(f"    - {lbl}: ({pt[0]:.1f}, {pt[1]:.1f})")

    unwarped, H_inv, (out_w, out_h) = apply_perspective_transform(img, ordered_corners)
    print(f"[*] Rectified canvas dimensions: {out_w} x {out_h} pixels")
    print(f"[*] Inverse Homography Matrix (Source Slanted -> Destination Flat):\n{np.array2string(H_inv, precision=4, suppress_small=True)}")

    annotated = img.copy()
    pts_int = ordered_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    for idx, (pt, lbl, col) in enumerate(zip(ordered_corners, labels, colors)):
        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 8, col, -1, cv2.LINE_AA)
        cv2.putText(annotated, lbl, (int(pt[0]) - 40, int(pt[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_dir, "q2_unwarped_document.jpg"), unwarped)

    fig = plt.figure(figsize=(18, 7))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax1.set_title("1. Tilted Camera Capture with Detected Corners\n(Source Quadrilateral ROI)", fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"2. Inverse Perspective Transformation (Unwarped)\nRectified Output ({out_w} x {out_h} px)", fontsize=11, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    diag = np.full((500, 600, 3), 250, dtype=np.uint8)
    cv2.rectangle(diag, (40, 40), (560, 460), (230, 230, 230), -1)
    cv2.putText(diag, "HOMOGRAPHY DUALITY", (160, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(diag, "Forward Transform: H", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 50, 40), 2)
    cv2.putText(diag, "Flat Frontal -> Slanted Scene", (70, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    cv2.putText(diag, "Inverse Transform: H^-1", (70, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 120, 30), 2)
    cv2.putText(diag, "Slanted Scene -> Flat Rectified Document", (70, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    cv2.putText(diag, "Relation: H_inv @ H_fwd = I_3x3", (70, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 150), 2)
    cv2.putText(diag, f"det(H) = {np.linalg.det(H_inv):.4f}", (70, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    ax3.imshow(cv2.cvtColor(diag, cv2.COLOR_BGR2RGB))
    ax3.set_title("3. Mathematical Duality\n(Forward vs Inverse Homography)", fontsize=11, fontweight='bold')
    ax3.axis('off')

    plt.suptitle("Q2: Perspective Correction (Unwarping) Analysis\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_fig = os.path.join(output_dir, "q2_perspective_correction.png")
    plt.savefig(output_fig, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  -> Saved perspective correction result: {output_fig}")
    print("[SUCCESS] Task 2 completed successfully.")
    return output_fig

if __name__ == "__main__":
    run_perspective_correction()
