import cv2
import numpy as np

def enhance_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # 1. Histogram Equalization
    eq = cv2.equalizeHist(gray)
    
    # 2. Pseudocolor jet heatmap
    heat = cv2.applyColorMap(eq, cv2.COLORMAP_JET)
    
    # 3. Gray-World color balance
    b, g, r = cv2.split(heat)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    avg = (mb + mg + mr) / 3.0
    b_cb = np.clip(b * (avg / mb), 0, 255).astype(np.uint8)
    g_cb = np.clip(g * (avg / mg), 0, 255).astype(np.uint8)
    r_cb = np.clip(r * (avg / mr), 0, 255).astype(np.uint8)
    cb = cv2.merge([b_cb, g_cb, r_cb])
    
    # 4. Logarithmic transformation
    c_log = 255.0 / np.log(1.0 + np.max(cb))
    log_f = (c_log * np.log(1.0 + cb.astype(np.float32))).astype(np.uint8)
    
    # 5. Power-law transformation (gamma = 1.35 to suppress probe backscatter noise)
    gamma = 1.35
    enh = np.clip(255.0 * ((log_f / 255.0) ** gamma), 0, 255).astype(np.uint8)
    return enh

def main():
    cap = cv2.VideoCapture('data/echocardiogram.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33
    
    print('Starting Real-Time Echocardiogram Enhancement Stream...')
    print('Press [Q] in the display window to exit.')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop playback
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        enh = enhance_frame(frame)
        side_by_side = np.hstack([frame, enh])
        
        cv2.putText(side_by_side, 'RAW FEED', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(side_by_side, 'ENHANCED FEED', (frame.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Real-Time Echocardiogram Analysis (Press Q to quit)', side_by_side)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print('Stream closed.')

if __name__ == '__main__':
    main()
