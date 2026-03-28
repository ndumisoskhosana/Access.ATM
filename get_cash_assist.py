import cv2
import pyttsx3
from ultralytics import YOLO

def run_get_cash_assist():
    """
    Identifies the cash dispenser slot (typically below the keypad) 
    and guides the user via audio to retrieve their Rands.
    """
    # ---------------------------
    # INIT COMPONENTS
    # ---------------------------
    engine = pyttsx3.init()
    # Using the Nano model for speed; looks for 'keyboard' as an anchor
    model = YOLO("yolov8n.pt") 
    
    def speak(text):
        print("[AUDIO]:", text)
        engine.say(text)
        engine.runAndWait()

    cap = cv2.VideoCapture(0)
    
    # Initial instruction for the user
    speak("Navigating to the cash dispenser. Please move your hand below the keypad area.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        cx_frame, cy_frame = w // 2, h // 2
        
        results = model(frame, verbose=False)
        keypad_box = None

        # --- ANCHOR LOGIC ---
        # We find the keypad because the cash slot is physically tied to its position.
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                # YOLO COCO labels 'keyboard' or 'remote' often trigger for ATM keypads
                if label in ["keyboard", "remote"]:
                    keypad_box = list(map(int, box.xyxy[0]))

        if keypad_box:
            kx1, ky1, kx2, ky2 = keypad_box
            
            # Target X is the horizontal center of the keypad
            target_x = (kx1 + kx2) // 2
            
            # Target Y: In most SA ATMs (Nedbank, FNB, Standard Bank), 
            # the cash slot is roughly 10-20cm directly below the keypad.
            # We add an offset to the bottom coordinate (ky2) of the keypad box.
            target_y = ky2 + 120 

            # Visual Feedback for the Developer
            cv2.rectangle(frame, (kx1, ky1), (kx2, ky2), (255, 0, 0), 2)
            cv2.circle(frame, (target_x, target_y), 15, (0, 255, 0), -1)
            cv2.putText(frame, "CASH DISPENSER TARGET", (target_x - 80, target_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- GUIDANCE CALCULATIONS ---
            dx = target_x - cx_frame
            dy = target_y - cy_frame
            
            # Precision threshold (how close we need to be to say 'reached')
            threshold = 60

            if abs(dx) < threshold and abs(dy) < threshold:
                speak("Cash dispenser reached. You can now feel for your Rands in the slot.")
                speak("Thank you for banking. Please ensure you have all your notes.")
                
                cap.release()
                cv2.destroyAllWindows()
                return True # Signal to main script that cash is taken

            # --- DIRECTIONAL AUDIO ---
            # Horizontal Correction
            if dx > threshold:
                speak("Move right")
            elif dx < -threshold:
                speak("Move left")

            # Vertical Correction
            if dy > threshold:
                speak("Move down")
            elif dy < -threshold:
                speak("Move up")

        # Fallback if keypad is lost from view
        else:
            cv2.putText(frame, "Searching for keypad anchor...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Cash Dispenser Assist", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

if __name__ == "__main__":
    run_get_cash_assist()