import cv2
import pyttsx3
import speech_recognition as sr
import re
import time
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from ultralytics import YOLO

# Importing your custom helper modules
from card_slot_assist import run_card_slot_assist
from keypad_assist import run_keypad_assist
from get_cash_assist import run_get_cash_assist

# --- CONFIGURATION ---
AZURE_ENDPOINT = "YOUR_AZURE_ENDPOINT"
AZURE_KEY = "YOUR_AZURE_KEY"

def read_screen_with_azure(frame, client):
    """Sends current frame to Azure to read SA ATM screen text."""
    _, buffer = cv2.imencode('.jpg', frame)
    image_data = buffer.tobytes()
    try:
        result = client.analyze(image_data=image_data, visual_features=[VisualFeatures.READ])
        text_found = ""
        if result.read:
            for block in result.read.blocks:
                for line in block.lines:
                    text_found += line.text.lower() + " "
        return text_found
    except:
        return ""

def get_validated_voice_choice(options_list, engine, state_context=None):
    """
    Captures user speech and validates it.
    Handles the specific exception for Option 2 (View Balance).
    """
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print(f"[LISTENING for: {options_list}]...")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                choice = recognizer.recognize_google(audio).lower()
                
                # --- EXCEPTION: OPTION 2 NOT AVAILABLE ---
                if state_context == "SELECT_TRANSACTION":
                    if "2" in choice or "balance" in choice:
                        engine.say("Sorry, navigation option is not available as yet. Please try again.")
                        engine.runAndWait()
                        continue 

                # Check for valid "Get Cash" or other options
                for opt in options_list:
                    if opt in choice:
                        return opt
                
                # General invalid input
                engine.say("Sorry, that option is not available. Please try again.")
                engine.runAndWait()
                
            except sr.UnknownValueError:
                engine.say("I didn't catch that. Please try again.")
                engine.runAndWait()
            except sr.WaitTimeoutError:
                engine.say("I'm waiting for your choice. Please say an option.")
                engine.runAndWait()

def guide_to_side_button(frame, screen_box, button_index):
    """Calculates coordinates for physical buttons on the left bezel."""
    if not screen_box: return False
    sx1, sy1, sx2, sy2 = screen_box
    h, w, _ = frame.shape
    
    target_x = sx1 - 60
    screen_h = sy2 - sy1
    offsets = [0.2, 0.4, 0.6, 0.8] 
    target_y = sy1 + int(screen_h * offsets[button_index - 1])

    cx, cy = w // 2, h // 2
    return abs(target_x - cx) < 65 and abs(target_y - cy) < 65

def main():
    engine = pyttsx3.init()
    client = ImageAnalysisClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))
    model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(0)
    state = "INSERT_CARD"
    screen_box = None
    last_atm_seen_time = time.time()
    
    def speak(text):
        print(f"[ASSISTANT]: {text}")
        engine.say(text)
        engine.runAndWait()

    speak("ATM Assistant started. Please point the camera at the machine.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- EXCEPTION 1: ATM VISIBILITY CHECK ---
        results = model(frame, verbose=False)
        atm_in_view = False
        for r in results:
            for box in r.boxes:
                if model.names[int(box.cls[0])] in ["tv", "laptop"]:
                    screen_box = list(map(int, box.xyxy[0]))
                    atm_in_view = True
                    last_atm_seen_time = time.time()

        if not atm_in_view and (time.time() - last_atm_seen_time) > 4:
            cv2.putText(frame, "ATM NOT FOUND", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            speak("ATM not found. Please point the camera back at the screen.")
            last_atm_seen_time = time.time()
            continue

        # --- TRANSACTION STATE MACHINE ---

        if state == "INSERT_CARD":
            if run_card_slot_assist(): 
                state = "CHECK_PIN"

        elif state == "CHECK_PIN":
            if "pin" in read_screen_with_azure(frame, client):
                speak("PIN requested. Moving to keypad.")
                if run_keypad_assist():
                    state = "SELECT_TRANSACTION"

        elif state == "SELECT_TRANSACTION":
            text = read_screen_with_azure(frame, client)
            if "transaction" in text or "choose" in text:
                speak("Transaction menu. Say 1 for Get Cash or 2 for View Balance.")
                choice = get_validated_voice_choice(["1", "cash"], engine, state_context="SELECT_TRANSACTION")
                
                if "1" in choice or "cash" in choice:
                    speak("Get Cash selected. Move to the first button on the top left.")
                    if guide_to_side_button(frame, screen_box, 1):
                        state = "SELECT_ACCOUNT"

        elif state == "SELECT_ACCOUNT":
            if "account" in read_screen_with_azure(frame, client):
                speak("Say 1 for Savings, 2 for Credit, or 3 for Checking.")
                choice = get_validated_voice_choice(["1", "2", "3", "savings", "credit", "checking"], engine)
                mapping = {"1": 1, "savings": 1, "2": 2, "credit": 2, "3": 3, "checking": 3}
                if guide_to_side_button(frame, screen_box, mapping.get(choice, 1)):
                    state = "WITHDRAWAL_AMOUNT"

        elif state == "WITHDRAWAL_AMOUNT":
            if "withdrawal" in read_screen_with_azure(frame, client):
                speak("Moving to 'Own Amount' button at the bottom left.")
                if guide_to_side_button(frame, screen_box, 4):
                    state = "ENTER_RAND_VALUE"

        elif state == "ENTER_RAND_VALUE":
            speak("Enter your Rand amount on the keypad.")
            if run_keypad_assist():
                content = read_screen_with_azure(frame, client)
                amount = re.search(r'r\s?(\d+)', content)
                amt_text = amount.group(0) if amount else "the amount shown"
                speak(f"The screen shows {amt_text}. Say 'Yes' to confirm.")
                
                if "yes" in get_validated_voice_choice(["yes"], engine):
                    speak("Confirmed. Press the green Enter key.")
                    state = "GET_CASH"

        elif state == "GET_CASH":
            text = read_screen_with_azure(frame, client)
            if "dispensing" in text or "take" in text:
                if run_get_cash_assist():
                    state = "TAKE_CARD"

        elif state == "TAKE_CARD":
            speak("Transaction complete. Returning to card slot for retrieval.")
            if run_card_slot_assist():
                speak("Thank you for banking. Have a safe day!")
                break

        cv2.imshow("Main ATM Assist", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    cap.release()