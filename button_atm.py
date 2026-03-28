import cv2
import time
import pytesseract
import pyttsx3

from card_slot_assist import run_card_slot_assist


# ---------------------------
# SIMPLE BUILT-IN TTS
# ---------------------------
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()


def detect_enter_pin(frame):
    """
    Detects 'Enter PIN' text on ATM screen using OCR.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    text = pytesseract.image_to_string(gray)

    if text:
        text = text.lower()
        if "enter pin" in text or "pin" in text:
            return True

    return False


def run_atm_button_flow(camera_index=0):
    """
    Flow:
    1. Prompt user to get closer
    2. Use card_slot_assist to guide card insertion
    3. Detect 'Enter PIN' on screen
    """

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        speak("Camera not accessible")
        return

    # ---------------------------
    # STEP 1: Prompt user
    # ---------------------------
    speak("Please move closer to the ATM")
    time.sleep(3)

    # ---------------------------
    # STEP 2: Card slot navigation
    # ---------------------------
    speak("Locating card slot")

    inserted = run_card_slot_assist(cap)

    if not inserted:
        speak("Card insertion failed")
        cap.release()
        return

    speak("Card inserted successfully")

    # ---------------------------
    # STEP 3: Detect PIN screen
    # ---------------------------
    speak("Waiting for PIN screen")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if detect_enter_pin(frame):
            speak("Enter PIN screen detected")
            break

    # ---------------------------
    # NEXT STEP (NOT CALLED YET)
    # ---------------------------
    # guide_to_keypad(cap)

    cap.release()