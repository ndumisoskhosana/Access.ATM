import cv2
import pyttsx3
import os
import time
from dotenv import load_dotenv
from ultralytics import YOLO

# Modern Azure Vision Imports
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# ---------------------------
# SECURE CONFIGURATION
# ---------------------------
load_dotenv()
endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
key = os.environ.get("AZURE_VISION_KEY")

if not endpoint or not key:
    raise ValueError("Missing Azure keys! Check your .env file.")

# Initialize the Azure Client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# ---------------------------
# INIT TEXT TO SPEECH & YOLO POSE
# ---------------------------
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate + 20)

def speak(text):
    print(f"[AUDIO]: {text}")
    engine.say(text)
    engine.runAndWait()

# Load YOLOv8 Pose (Will auto-download the tiny model on first run)
print("[INFO] Loading YOLOv8 Pose Model...")
pose_model = YOLO("yolov8n-pose.pt")

# ---------------------------
# LIVE AZURE OCR FUNCTION
# ---------------------------
def get_text_bounding_boxes_from_azure(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return {}
    
    image_bytes = encoded_image.tobytes()
    boxes = {}

    try:
        result = client.analyze(
            image_data=image_bytes,
            visual_features=[VisualFeatures.READ]
        )

        if result.read is not None:
            for line in result.read.blocks[0].lines:
                for word in line.words:
                    pts = word.bounding_polygon
                    xs = [p.x for p in pts]
                    ys = [p.y for p in pts]
                    
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    clean_text = word.text.strip().lower()
                    boxes[clean_text] = [x_min, y_min, x_max, y_max]
                    
    except Exception as e:
        print(f"[AZURE ERROR]: {e}")

    return boxes

# ---------------------------
# WRIST GUIDANCE FUNCTION
# ---------------------------
def guide_hand_to_text(target_box, hand_x, hand_y):
    tx1, ty1, tx2, ty2 = target_box
    target_cx = (tx1 + tx2) // 2
    target_cy = (ty1 + ty2) // 2

    dx = target_cx - hand_x
    dy = target_cy - hand_y
    
    threshold = 60 # Pixel leeway

    if abs(dx) < threshold and abs(dy) < threshold:
        speak("You are on the button. Tap now.")
        return True

    if dx > threshold:
        speak("Move right")
    elif dx < -threshold:
        speak("Move left")

    if dy > threshold:
        speak("Move down")
    elif dy < -threshold:
        speak("Move up")

    return False

# ---------------------------
# MAIN LOOP & STATE MACHINE
# ---------------------------
def run_touchscreen_assist():
    cap = cv2.VideoCapture(0)
    
    current_state = "LOCATING_TARGET"
    target_word = "savings" 
    target_box = None
    
    # Cooldown variables
    last_scan_time = 0
    scan_cooldown = 3.0 # Wait 3 seconds between Azure API calls
    
    speak(f"System active. Show me the ATM screen. Looking for the {target_word} option.")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # 1. TRIGGER AZURE OCR (WITH COOLDOWN)
        if target_box is None and current_state == "LOCATING_TARGET":
            current_time = time.time()
            
            # Only hit the API if 3 seconds have passed
            if current_time - last_scan_time > scan_cooldown:
                print("[INFO] Scanning screen with Azure...")
                boxes = get_text_bounding_boxes_from_azure(frame)
                last_scan_time = current_time # Reset the timer
                
                if target_word in boxes:
                    target_box = boxes[target_word]
                    speak(f"Located {target_word}. Show me your hand.")
                    current_state = "GUIDING_HAND"
            
            # Keep updating the video feed smoothly while waiting
            if target_box is None:
                cv2.putText(frame, f"Looking for '{target_word}'... Hold still.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 2. HAND TRACKING (USING YOLO POSE)
        if current_state == "GUIDING_HAND" and target_box:
            tx1, ty1, tx2, ty2 = target_box
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
            cv2.putText(frame, "TARGET", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            results = pose_model(frame, verbose=False)
            
            for r in results:
                if r.keypoints and len(r.keypoints.xy) > 0:
                    kpts = r.keypoints.xy[0] 
                    
                    if len(kpts) >= 11:
                        lx, ly = int(kpts[9][0]), int(kpts[9][1])
                        rx, ry = int(kpts[10][0]), int(kpts[10][1])
                        
                        hand_x, hand_y = 0, 0
                        
                        if rx > 0 and ry > 0:
                            hand_x, hand_y = rx, ry
                            cv2.circle(frame, (rx, ry), 15, (255, 0, 0), -1) 
                        elif lx > 0 and ly > 0:
                            hand_x, hand_y = lx, ly
                            cv2.circle(frame, (lx, ly), 15, (0, 255, 255), -1) 
                            
                        if hand_x > 0:
                            touched = guide_hand_to_text(target_box, hand_x, hand_y)
                            if touched:
                                current_state = "SUCCESS"
                                target_box = None
                                speak("Transaction selected successfully.")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'): 
            current_state = "LOCATING_TARGET"
            target_box = None
            last_scan_time = 0 # Reset timer so it scans immediately
            speak("Resetting scan.")
        elif key == 27: 
            break

        cv2.imshow("Isazi Hackathon - Touchscreen Assist", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run_touchscreen_assist()
    except Exception as e:
        print(f"[CRITICAL ERROR]: {e}")