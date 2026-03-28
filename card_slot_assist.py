# card_slot_assist.py
import cv2
import pyttsx3
from ultralytics import YOLO


def run_card_slot_assist():
    """
    Detects a card insertion slot relative to ATM screen and keypad,
    guides the user via audio until aligned with the slot.
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
    def guide_to_slot(frame, box):
        h, w, _ = frame.shape

        cx_frame = w // 2
        cy_frame = h // 2

        x1, y1, x2, y2 = box
        cx_slot = (x1 + x2) // 2
        cy_slot = (y1 + y2) // 2

        dx = cx_slot - cx_frame
        dy = cy_slot - cy_frame

        threshold = 60

        if abs(dx) < threshold and abs(dy) < threshold:
            speak("Card slot centered. You can insert your card.")
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
    # CAMERA LOOP
    # ---------------------------
    cap = cv2.VideoCapture(0)

    aligned = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        screen_box = None
        keypad_box = None
        slot_box = None

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Debug visuals
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Approximate detections
                if label in ["tv", "laptop"]:
                    screen_box = (x1, y1, x2, y2)

                elif label in ["keyboard", "remote"]:
                    keypad_box = (x1, y1, x2, y2)

                # Heuristic: thin horizontal object = possible slot
                elif label in ["cell phone", "book"]:
                    slot_box = (x1, y1, x2, y2)

        # ---------------------------
        # POSITION LOGIC
        # ---------------------------
        if screen_box and slot_box:
            _, _, _, sy2 = screen_box
            sx1, sy1, sx2, _ = screen_box

            x1, y1, x2, y2 = slot_box

            # Slot should be BELOW screen
            if y1 > sy2:

                # Optional: prefer right side of screen
                cx_slot = (x1 + x2) // 2
                cx_screen = (sx1 + sx2) // 2

                if cx_slot >= cx_screen:  # right side bias
                    aligned = guide_to_slot(frame, slot_box)

        if aligned:
            speak("Insert your card carefully.")
            aligned = False  # prevent repeating constantly

        cv2.imshow("Card Slot Assist", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()