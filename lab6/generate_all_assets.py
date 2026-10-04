"""
Asset Generation Script for Lab 06:
Generates synthetic and realistic images/videos for all 8 lab tasks.
"""
import cv2
import numpy as np
import os

os.makedirs("assets", exist_ok=True)

# -------------------------------------------------------------
# 0. Foundation Scene: edges_input.jpg (Architectural structure)
# -------------------------------------------------------------
def make_edges_input():
    img = np.ones((500, 700, 3), dtype=np.uint8) * 220
    # Sky gradient
    for y in range(250):
        img[y, :] = [240 - y//3, 220 - y//4, 180 - y//5]
    # Building body
    cv2.rectangle(img, (100, 200), (600, 480), (140, 150, 160), -1)
    cv2.rectangle(img, (100, 200), (600, 480), (40, 40, 40), 3)
    # Roof (triangle)
    roof = np.array([[80, 200], [350, 70], [620, 200]], np.int32)
    cv2.fillPoly(img, [roof], (90, 80, 150))
    cv2.polylines(img, [roof], True, (30, 20, 60), 3)
    # Windows grid
    for r in range(3):
        for c in range(5):
            wx = 140 + c * 90
            wy = 230 + r * 75
            cv2.rectangle(img, (wx, wy), (wx + 60, wy + 50), (220, 230, 240), -1)
            cv2.rectangle(img, (wx, wy), (wx + 60, wy + 50), (30, 30, 30), 2)
            cv2.line(img, (wx + 30, wy), (wx + 30, wy + 50), (30, 30, 30), 1)
            cv2.line(img, (wx, wy + 25), (wx + 60, wy + 25), (30, 30, 30), 1)
    # Door
    cv2.rectangle(img, (310, 370), (390, 480), (50, 70, 100), -1)
    cv2.rectangle(img, (310, 370), (390, 480), (20, 20, 20), 2)
    # Add subtle texture noise
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite("assets/edges_input.jpg", img)
    print("Created assets/edges_input.jpg")

# -------------------------------------------------------------
# 1. Task 1: Computer Lab Screens (lab_screens.jpg)
# -------------------------------------------------------------
def make_lab_screens():
    # 6 stations in 2 rows. 3 ON, 2 OFF, 1 MISSING
    w, h = 900, 600
    img = np.ones((h, w, 3), dtype=np.uint8) * 190
    # Background wall and desks
    cv2.rectangle(img, (0, 0), (w, 180), (160, 170, 180), -1)
    # Desk 1
    cv2.rectangle(img, (40, 240), (860, 300), (90, 95, 100), -1)
    # Desk 2
    cv2.rectangle(img, (40, 480), (860, 560), (70, 75, 80), -1)

    # Monitor specs: (center_x, center_y, width, height, state: 'ON' | 'OFF' | 'MISSING')
    stations = [
        # Row 1 (back row)
        (170, 170, 140, 90, 'ON'),
        (450, 170, 140, 90, 'OFF'),
        (730, 170, 140, 90, 'ON'),
        # Row 2 (front row)
        (170, 400, 160, 105, 'OFF'),
        (450, 400, 160, 105, 'ON'),
        (730, 400, 160, 105, 'MISSING')
    ]

    for cx, cy, mw, mh, state in stations:
        x1, y1 = cx - mw//2, cy - mh//2
        x2, y2 = cx + mw//2, cy + mh//2
        if state == 'MISSING':
            # Draw empty stand / desk placeholder
            cv2.rectangle(img, (cx - 20, y2 + 10), (cx + 20, y2 + 20), (50, 50, 50), -1)
            cv2.putText(img, "EMPTY DESK", (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
            continue

        # Draw monitor stand
        cv2.rectangle(img, (cx - 8, y2), (cx + 8, y2 + 15), (40, 40, 40), -1)
        cv2.rectangle(img, (cx - 25, y2 + 15), (cx + 25, y2 + 20), (30, 30, 30), -1)

        # Draw outer bezel
        cv2.rectangle(img, (x1, y1), (x2, y2), (25, 25, 25), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (10, 10, 10), 2)

        # Screen display area (inner border)
        bx1, by1 = x1 + 6, y1 + 6
        bx2, by2 = x2 - 6, y2 - 6
        if state == 'ON':
            # Lit screen (bright blue/cyan desktop with some windows)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (220, 180, 100), -1)
            # window on screen
            cv2.rectangle(img, (bx1 + 10, by1 + 10), (bx1 + 70, by1 + 45), (255, 255, 255), -1)
            cv2.putText(img, "SYS:OK", (bx1 + 15, by1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        else:
            # OFF screen (dark gray)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (35, 35, 35), -1)

    cv2.imwrite("assets/lab_screens.jpg", img)
    print("Created assets/lab_screens.jpg")

# -------------------------------------------------------------
# 2. Task 2: Asset Tracking (Catalogue & Lab Scene)
# -------------------------------------------------------------
def make_asset_tracking_data():
    # Helper to add texture details so SIFT finds lots of keypoints
    def add_sift_texture(im, label):
        cv2.putText(im, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Add QR-code-like or logo pattern
        for i in range(5):
            for j in range(5):
                if (i * 3 + j * 7) % 2 == 0:
                    cv2.rectangle(im, (20 + i*12, 50 + j*12), (30 + i*12, 60 + j*12), (200, 220, 250), -1)
        # Barcode
        for k in range(15):
            bx = 100 + k * 5
            thick = 1 if k % 3 == 0 else 2
            cv2.line(im, (bx, 50), (bx, 100), (240, 240, 240), thick)

    # 1. Monitor Reference (160x120)
    mon = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(mon, (10, 10), (150, 100), (60, 60, 60), -1)
    cv2.rectangle(mon, (18, 18), (142, 92), (180, 140, 90), -1)
    add_sift_texture(mon, "MONITOR")
    cv2.imwrite("assets/ref_monitor.jpg", mon)

    # 2. Keyboard Reference (80x180)
    kb = np.zeros((80, 180, 3), dtype=np.uint8)
    cv2.rectangle(kb, (5, 5), (175, 75), (40, 40, 40), -1)
    # Draw key grid
    for r in range(4):
        for c in range(12):
            kx = 15 + c * 13
            ky = 12 + r * 15
            cv2.rectangle(kb, (kx, ky), (kx + 10, ky + 12), (190, 190, 190), -1)
    cv2.putText(kb, "KB-FAST", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.imwrite("assets/ref_keyboard.jpg", kb)

    # 3. CPU Tower Reference (160x100)
    tow = np.zeros((160, 100, 3), dtype=np.uint8)
    cv2.rectangle(tow, (8, 8), (92, 152), (50, 50, 50), -1)
    # Power button & vents
    cv2.circle(tow, (50, 30), 8, (0, 255, 255), -1)
    cv2.circle(tow, (50, 30), 12, (200, 200, 200), 2)
    for v in range(6):
        cv2.line(tow, (25, 60 + v * 10), (75, 60 + v * 10), (220, 220, 220), 2)
    add_sift_texture(tow, "TOWER")
    cv2.imwrite("assets/ref_tower.jpg", tow)

    # 4. Lab Scene containing these assets placed with affine/perspective transforms
    scene = np.ones((600, 800, 3), dtype=np.uint8) * 160
    # Table surface
    cv2.rectangle(scene, (50, 200), (750, 560), (110, 100, 95), -1)
    cv2.rectangle(scene, (50, 200), (750, 560), (60, 50, 45), 2)

    # Place Monitor at (120, 240), slightly scaled
    mon_scaled = cv2.resize(mon, (200, 150))
    scene[240:240+150, 120:120+200] = mon_scaled

    # Place Keyboard at (150, 420)
    kb_scaled = cv2.resize(kb, (220, 95))
    scene[420:420+95, 150:150+220] = kb_scaled

    # Place Tower at (480, 220)
    tow_scaled = cv2.resize(tow, (120, 190))
    scene[220:220+190, 480:480+120] = tow_scaled

    # Add background lab room details
    cv2.putText(scene, "LAB ASSET STATION #3", (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    cv2.imwrite("assets/lab_scene.jpg", scene)
    print("Created assets for Task 2 (Asset tracking catalogue and scene)")

# -------------------------------------------------------------
# 3. Task 4: Video Object Recognition (object.jpg & test_video.mp4)
# -------------------------------------------------------------
def make_video_object_data():
    # Reference object: rich textured book cover / label (160x220)
    ref = np.zeros((220, 160, 3), dtype=np.uint8)
    # Background pattern
    for y in range(220):
        ref[y, :] = [(y * 2) % 255, 100, 200 - (y % 180)]
    cv2.rectangle(ref, (10, 10), (150, 210), (255, 255, 255), 3)
    cv2.circle(ref, (80, 80), 35, (0, 255, 255), -1)
    cv2.circle(ref, (80, 80), 25, (255, 0, 100), -1)
    cv2.putText(ref, "OPENCV", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(ref, "LAB 6", (45, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite("assets/object.jpg", ref)

    # Generate 50-frame video where object moves, rotates, and scales in a scene
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("assets/test_video.mp4", fourcc, 20.0, (640, 480))
    ref_h, ref_w = ref.shape[:2]

    for f in range(50):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
        # Background office desk and wall
        cv2.rectangle(frame, (0, 300), (640, 480), (120, 120, 130), -1)
        cv2.line(frame, (0, 300), (640, 300), (50, 50, 60), 2)
        cv2.putText(frame, f"SURVEILLANCE CAM - FRAME {f:02d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)

        # Animate object: trajectory
        t = f / 50.0
        scale = 0.8 + 0.3 * np.sin(np.pi * t)
        angle = -15 + 30 * t
        tx = int(120 + 320 * t)
        ty = int(150 + 70 * np.sin(2 * np.pi * t))

        # Affine transform for object
        M = cv2.getRotationMatrix2D((ref_w/2, ref_h/2), angle, scale)
        M[0, 2] += (tx - ref_w/2)
        M[1, 2] += (ty - ref_h/2)
        warped = cv2.warpAffine(ref, M, (640, 480), borderValue=(0, 0, 0))

        # Mask object onto frame
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = gray_w > 0
        frame[mask] = warped[mask]

        out.write(frame)

    out.release()
    print("Created assets/object.jpg and assets/test_video.mp4")

# -------------------------------------------------------------
# 4. Task 5: Panoramic Images (pan1.jpg, pan2.jpg, pan3.jpg)
# -------------------------------------------------------------
def make_panorama_data():
    # Create a wide master panoramic canvas (1200x500) with rich buildings, mountains, trees, features
    wide = np.ones((500, 1200, 3), dtype=np.uint8) * 230
    # Sky gradient
    for y in range(250):
        wide[y, :] = [250 - y//3, 220 - y//4, 180 - y//5]
    # Mountains in background
    pts_mountains = np.array([[0, 240], [200, 120], [450, 220], [700, 100], [950, 210], [1200, 130], [1200, 350], [0, 350]], np.int32)
    cv2.fillPoly(wide, [pts_mountains], (130, 140, 120))

    # Foreground buildings and distinctive landmark signs
    landmarks = [
        (150, 220, 120, 180, (90, 100, 140), "CITY HALL"),
        (380, 200, 160, 200, (110, 80, 90), "TECH TOWER"),
        (650, 180, 140, 220, (80, 120, 100), "FAST NUCES"),
        (920, 210, 150, 190, (130, 110, 90), "LIBRARY")
    ]
    for lx, ly, lw, lh, color, text in landmarks:
        cv2.rectangle(wide, (lx, ly), (lx + lw, ly + lh), color, -1)
        cv2.rectangle(wide, (lx, ly), (lx + lw, ly + lh), (20, 20, 20), 2)
        # Windows
        for wy in range(ly + 20, ly + lh - 30, 30):
            for wx in range(lx + 15, lx + lw - 25, 25):
                cv2.rectangle(wide, (wx, wy), (wx + 15, wy + 15), (255, 245, 200), -1)
                cv2.rectangle(wide, (wx, wy), (wx + 15, wy + 15), (30, 30, 30), 1)
        # Text sign
        cv2.putText(wide, text, (lx + 8, ly + lh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Road and ground
    cv2.rectangle(wide, (0, 380), (1200, 500), (70, 70, 75), -1)
    for d in range(0, 1200, 60):
        cv2.rectangle(wide, (d, 435), (d + 35, 445), (255, 255, 255), -1)

    # Add distinctive trees with texture
    for tx in [70, 320, 570, 840, 1120]:
        cv2.rectangle(wide, (tx - 5, 340), (tx + 5, 390), (40, 60, 90), -1)
        cv2.circle(wide, (tx, 330), 25, (40, 130, 50), -1)
        cv2.circle(wide, (tx - 10, 340), 18, (30, 110, 40), -1)
        cv2.circle(wide, (tx + 10, 340), 18, (50, 140, 60), -1)

    # Crop 3 overlapping views: 0..500, 320..820, 640..1140 (each 500px wide, ~180px overlap)
    pan1 = wide[:, 0:520]
    pan2 = wide[:, 320:840]
    pan3 = wide[:, 650:1170]

    cv2.imwrite("assets/pan1.jpg", pan1)
    cv2.imwrite("assets/pan2.jpg", pan2)
    cv2.imwrite("assets/pan3.jpg", pan3)
    print("Created assets/pan1.jpg, pan2.jpg, pan3.jpg")

# -------------------------------------------------------------
# 5. Task 6: Road Lane Image (road.jpg)
# -------------------------------------------------------------
def make_road_data():
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 100
    # Sky and horizon
    for y in range(int(h * 0.55)):
        img[y, :] = [230 - y//3, 190 - y//4, 140 - y//4]
    # Landscape hills
    cv2.rectangle(img, (0, int(h * 0.52)), (w, int(h * 0.58)), (80, 120, 80), -1)

    # Road polygon (vanishing point at horizon)
    road_pts = np.array([
        [0, h],
        [int(w * 0.45), int(h * 0.58)],
        [int(w * 0.55), int(h * 0.58)],
        [w, h]
    ], np.int32)
    cv2.fillPoly(img, [road_pts], (60, 60, 65))

    # Left solid lane marking (white/yellow)
    left_lane = np.array([
        [int(w * 0.15), h],
        [int(w * 0.18), h],
        [int(w * 0.47), int(h * 0.58)],
        [int(w * 0.46), int(h * 0.58)]
    ], np.int32)
    cv2.fillPoly(img, [left_lane], (235, 235, 245))

    # Right dashed lane marking
    for d in [0.92, 0.80, 0.70, 0.63]:
        y_bottom = int(h * d)
        y_top = int(h * (d - 0.05))
        # interpolate x along line from (0.85*w, h) to (0.53*w, 0.58*h)
        alpha_b = (y_bottom - h * 0.58) / (h * 0.42)
        alpha_t = (y_top - h * 0.58) / (h * 0.42)
        xb = int(w * 0.53 + alpha_b * (w * 0.85 - w * 0.53))
        xt = int(w * 0.53 + alpha_t * (w * 0.85 - w * 0.53))
        thick = max(3, int(10 * alpha_b))
        cv2.line(img, (xb, y_bottom), (xt, y_top), (240, 240, 245), thick)

    cv2.imwrite("assets/road.jpg", img)
    print("Created assets/road.jpg")

# -------------------------------------------------------------
# 6. Task 8: Smart Security System Video (security_video.mp4)
# -------------------------------------------------------------
def make_security_video():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("assets/security_video.mp4", fourcc, 20.0, (640, 480))
    h, w = 480, 640

    # Defined zone: (200, 150) to (500, 400)
    for f in range(60):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 190
        # Background office desk / safe table
        cv2.rectangle(frame, (100, 100), (550, 420), (140, 135, 130), -1)
        cv2.rectangle(frame, (100, 100), (550, 420), (90, 85, 80), 2)
        cv2.putText(frame, f"SECURITY ZONE CAM - FRAME {f:02d}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

        # Baseline: frames 0 to 35 are EMPTY (only subtle sensor noise)
        # Frames 36 to 60: unauthorized suitcase/box enters zone
        if f >= 36:
            # Animate object entering zone
            prog = min(1.0, (f - 35) / 10.0)
            obj_x = int(120 + prog * 160) # moves to 280 (inside 200..500)
            obj_y = int(180 + prog * 60)  # moves to 240
            # Draw textured bag/briefcase with high internal edges
            cv2.rectangle(frame, (obj_x, obj_y), (obj_x + 120, obj_y + 90), (40, 40, 120), -1)
            cv2.rectangle(frame, (obj_x, obj_y), (obj_x + 120, obj_y + 90), (10, 10, 10), 3)
            # Bag straps and locks
            cv2.line(frame, (obj_x + 35, obj_y), (obj_x + 35, obj_y + 90), (220, 220, 220), 2)
            cv2.line(frame, (obj_x + 85, obj_y), (obj_x + 85, obj_y + 90), (220, 220, 220), 2)
            cv2.rectangle(frame, (obj_x + 30, obj_y + 40), (obj_x + 40, obj_y + 50), (255, 215, 0), -1)
            cv2.rectangle(frame, (obj_x + 80, obj_y + 40), (obj_x + 90, obj_y + 50), (255, 215, 0), -1)

        # Add camera sensor noise
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)

    out.release()
    print("Created assets/security_video.mp4")

if __name__ == "__main__":
    make_edges_input()
    make_lab_screens()
    make_asset_tracking_data()
    make_video_object_data()
    make_panorama_data()
    make_road_data()
    make_security_video()
    print("All assets successfully created in assets/ folder!")
