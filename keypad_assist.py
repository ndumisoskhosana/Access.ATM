# atm_keypad_assist.py

import cv2
import pyttsx3
from ultralytics import YOLO


def run_keypad_assist():
    """
    Detects a keypad under a screen, guides user via audio,
    and describes keypad layout once centered.
    """

    # ---------------------------
    # INIT TEXT TO SPEECH
    # ---------------------------
    engine = pyttsx3.init()

    def speak(text):
        print("[AUDIO]:", text)
        engine.say(text)
        engine.runAndWait()

    # ---------------------------
    # LOAD MODEL
    # ---------------------------
    model = YOLO("yolov8n.pt")

    # ---------------------------
    # GUIDANCE FUNCTION
    # ---------------------------
    def guide_to_keypad(frame, box):
        h, w, _ = frame.shape

        cx_frame = w // 2
        cy_frame = h // 2

        x1, y1, x2, y2 = box
        cx_key = (x1 + x2) // 2
        cy_key = (y1 + y2) // 2

        dx = cx_key - cx_frame
        dy = cy_key - cy_frame

        threshold = 60

        if abs(dx) < threshold and abs(dy) < threshold:
            speak("Keyboard centered.")
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
    # LAYOUT DESCRIPTION
    # ---------------------------
    def describe_layout():
        speak("Keypad detected.")
        speak("Top row: 1 2 3 and cancel button.")
        speak("Second row: 4 5 6 and clear button.")
        speak("Third row: 7 8 9 and enter button.")
        speak("Bottom row: zero in the center.")
        speak("There is an empty button below the clear button.")

    # ---------------------------
    # CAMERA LOOP
    # ---------------------------
    cap = cv2.VideoCapture(0)
    described = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        keypad_box = None
        screen_box = None

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Optional debug visuals
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                if label in ["keyboard", "remote"]:
                    keypad_box = (x1, y1, x2, y2)

                if label in ["tv", "laptop"]:
                    screen_box = (x1, y1, x2, y2)

        # Ensure keypad is below screen
        if keypad_box and screen_box:
            _, _, _, sy2 = screen_box
            _, ky1, _, _ = keypad_box

            if ky1 > sy2:
                centered = guide_to_keypad(frame, keypad_box)

                if centered and not described:
                    describe_layout()
                    described = True

        cv2.imshow("ATM Keypad Assist", frame)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()