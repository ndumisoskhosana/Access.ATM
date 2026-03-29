from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import urllib.request
from dotenv import load_dotenv
from pyngrok import ngrok
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# ==========================================
# MODERN MEDIAPIPE INITIALIZATION 
# ==========================================
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("[INFO] Checking for Modern MediaPipe Model...")
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("[INFO] Downloading model from Google (This only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

print("[INFO] Loading Modern MediaPipe Tasks API...")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
hands_detector = vision.HandLandmarker.create_from_options(options)

# ---------------------------
# INITIALIZATION & STATE
# ---------------------------
app = Flask(__name__)
load_dotenv()

endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
key = os.environ.get("AZURE_VISION_KEY")

if not endpoint or not key:
    raise ValueError("Missing Azure keys! Check your .env file.")

client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
print("[INFO] Server Initialized and Ready...")

current_state = "READING_SCREEN"
dynamic_menu = {} 
target_box = None

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def get_text_bounding_boxes_from_azure(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success: return {}
    image_bytes = encoded_image.tobytes()
    boxes = {}
    try:
        result = client.analyze(image_data=image_bytes, visual_features=[VisualFeatures.READ])
        if result.read is not None and len(result.read.blocks) > 0:
            for line in result.read.blocks[0].lines:
                text = line.text.strip().lower()
                pts = line.bounding_polygon
                xs = [p.x for p in pts]
                ys = [p.y for p in pts]
                boxes[text] = [min(xs), min(ys), max(xs), max(ys)]
    except Exception as e:
        print(f"[AZURE ERROR]: {e}")
    return boxes

def get_guidance_string(target_box, finger_x, finger_y):
    tx1, ty1, tx2, ty2 = target_box
    target_cx = (tx1 + tx2) // 2
    target_cy = (ty1 + ty2) // 2

    dx = target_cx - finger_x
    dy = target_cy - finger_y
    threshold = 35

    if abs(dx) < threshold and abs(dy) < threshold: 
        return "Target locked. Push your finger straight forward to tap the glass."
    
    if abs(dx) > abs(dy):
        if dx > threshold: return "Move right."
        else: return "Move left."
    else:
        if dy > threshold: return "Move down."
        else: return "Move up."

# ---------------------------
# ROUTES
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    global current_state, target_box, dynamic_menu
    
    requested_target = request.form.get('target') 
    
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    instruction = ""

    # ==========================================
    # GLOBAL INTENT & VOICE RECOGNITION 
    # ==========================================
    if requested_target:
        req = requested_target.lower()
        print(f"[VOICE INPUT] User said: '{req}'")
        
        # 1. THE NUCLEAR OVERRIDE: Catch numbers immediately, regardless of state!
        matched_num = None
        word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5"}
        
        for word, num in word_to_num.items():
            if word in req:
                matched_num = num
                break
                
        if matched_num and matched_num in dynamic_menu:
            target_box = dynamic_menu[matched_num]
            instruction = f"Option {matched_num} selected. Show me your hand."
            current_state = "GUIDING_HAND"
            return jsonify({"instruction": instruction, "state": current_state}) 

        # 2. STANDARD CONTEXT COMMANDS
        elif any(word in req for word in ["what", "read", "screen", "options", "repeat", "help", "ready", "next", "continue"]):
            current_state = "READING_SCREEN"
        elif any(word in req for word in ["cancel", "stop", "back", "reset"]):
            # THE FIX: Force it into a PAUSE state and immediately return the audio. 
            # This prevents it from instantly re-reading the screen and spamming the user.
            current_state = "PAUSED"
            target_box = None
            instruction = "Assistant reset. If you need to cancel the bank transaction, please press the physical cancel button on the metal keypad. Find the raised bump on the number 5 to orient your hand. The cancel button is found in the top right corner of the keypad. Tap your phone and say 'Ready' when you want me to scan the screen again."
            return jsonify({"instruction": instruction, "state": current_state}) # Returns immediately! 
        elif any(word in req for word in ["where", "keypad", "card", "cash", "slot", "physical"]):
            instruction = "The physical keypad is directly below the screen. Find the raised bump on the number 5 to orient your hand. The card slot is to the right of the keypad, and the cash dispenser is at the bottom of the machine. "
            return jsonify({"instruction": instruction, "state": current_state})

    # ==========================================
    # STATE 1: READ THE SCREEN
    # ==========================================
    if current_state == "READING_SCREEN":
        boxes = get_text_bounding_boxes_from_azure(frame)
        
        frame_height, frame_width, _ = frame.shape
        valid_boxes = {}
        for text, box in boxes.items():
            if box[3] <= (frame_height * 0.75):
                valid_boxes[text] = box
                
        screen_text_combined = " ".join(valid_boxes.keys())
        
        if any(phrase in screen_text_combined for phrase in ["card to begin", "awaiting card", "card insertion"]):
            instruction += "Welcome to the Bank of the Future. To find the card slot, feel the area to the right of the physical keypad below the screen. Please insert your card to begin."
            
        elif "pin" in screen_text_combined and ("enter" in screen_text_combined or "type" in screen_text_combined):
            instruction += "Screen asks for PIN. Move your hand directly below the center of the screen to find the physical keypad. Find the raised bump on the number 5 to orient your hand, type your PIN, and press the green enter button."
            
        elif any(word in screen_text_combined for word in ["take cash", "successful", "success", "complete", "thank you"]):
            instruction += "Transaction successful. Please take your card from the slot, and collect your cash or receipt from the dispensers below the keypad."
            
        elif any(word in screen_text_combined for word in ["declined", "failed", "error", "incorrect", "invalid"]):
            instruction += "An error or decline message is on the screen. To start over, tap your phone and say 'Cancel', or press the red physical cancel button on the keypad."
            
        elif "insert cash" in screen_text_combined or "insert notes" in screen_text_combined:
            instruction += "The cash deposit shutter is open. It is located below the keypad. Please insert your notes."
            
        else:
            dynamic_menu.clear()
            context_announcement = "" 
            audio_options = "Options are: "
            counter = 1
            added_texts = [] 
            
            sorted_texts = sorted(valid_boxes.keys(), key=lambda k: valid_boxes[k][1])
            
            for text in sorted_texts:
                box = valid_boxes[text]
                box_height = box[3] - box[1] 
                
                clean_text = ''.join(e for e in text if e.isalnum() or e.isspace()).strip()
                
                silent_ignore = ["simulator", "session", "select", "proceed", "menu", "bank", "keypad", "clear", "enter", "cancel"]
                if any(word in clean_text for word in silent_ignore) or clean_text.replace(" ", "").isdigit():
                    continue 

                context_phrases = ["current balance", "available balance", "user id", "welcome", "hello"]
                if any(word in clean_text for word in context_phrases):
                    if text not in context_announcement: 
                        context_announcement += f"{text}. "
                    continue 
                
                if len(clean_text.split()) <= 4 and len(clean_text) > 2 and box_height > 10: 
                    is_duplicate = False
                    for added in added_texts:
                        if clean_text in added or added in clean_text:
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        added_texts.append(clean_text)
                        dynamic_menu[str(counter)] = box
                        audio_options += f"{counter} for {clean_text}. "
                        counter += 1
            
            if counter > 1:
                instruction += context_announcement + audio_options + "Tap anywhere and say your number."
                current_state = "AWAITING_CHOICE" # THE FIX: Force the AI into a silent pause to wait for voice input!
            elif context_announcement:
                instruction += context_announcement + "Tap your phone and say 'Ready' to continue."
                current_state = "PAUSED"
            else:
                instruction += "Scanning screen..."

    # ==========================================
    # STATE 2: THE MENU WAITING ROOM
    # ==========================================
    elif current_state == "AWAITING_CHOICE":
        if requested_target:
            # If they spoke, but the "NUCLEAR OVERRIDE" at the top didn't catch a valid number
            instruction = "I didn't catch a valid number. You can say a number, or ask 'What are my options?'"
        else:
            # Absolute silence. The AI waits patiently without running OCR again.
            instruction = ""

    # ==========================================
    # STATE 3: MODERN MEDIAPIPE FINGERTIP TRACKING
    # ==========================================
    elif current_state == "GUIDING_HAND" and target_box:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = hands_detector.detect(mp_image)
        
        finger_x, finger_y = 0, 0
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            index_tip = hand_landmarks[8] 
            
            h, w, _ = frame.shape
            finger_x = int(index_tip.x * w)
            finger_y = int(index_tip.y * h) - 50

        if finger_x > 0:
            instruction = get_guidance_string(target_box, finger_x, finger_y) 
            
            if instruction == "Target locked. Push your finger straight forward to tap the glass.":
                instruction += " Once the next screen loads, tap your phone and say 'Ready'."
                current_state = "PAUSED" 
                target_box = None
        else:
            instruction = "Hand not visible. Please show your hand."
            
    # ==========================================
    # STATE 4: THE COOL-DOWN PAUSE
    # ==========================================
    elif current_state == "PAUSED":
        instruction = ""

    return jsonify({"instruction": instruction, "state": current_state})

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"\n==========\nOPEN THIS URL ON YOUR PHONE:\n{public_url}\n==========\n")
    app.run(host='0.0.0.0', port=5000)