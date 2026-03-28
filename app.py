from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from dotenv import load_dotenv
from pyngrok import ngrok
from ultralytics import YOLO
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# The Context-Aware Engine: Tells the AI what to do based on what screen is currently visible
ATM_NAVIGATOR = {
    "withdraw": {
        "welcome": "Action: Welcome. To find the card slot, feel the area to the right of the physical keypad below the screen. Insert your card to begin.",
        "insert card": "Action: The card slot is located to the right of the physical keypad below the screen. Please insert your card.",
        "enter pin": "Action: Move your hand directly below the center of the screen to find the physical keypad. Find the raised bump on the number 5 to orient your hand, type your PIN, and press the green enter button.",
        "withdrawal": "Touch", 
        "savings": "Touch",
        "500": "Touch", 
        "yes": "Touch",
        "take cash": "Action: Transaction complete. Move your hand to the bottom of the ATM machine, far below the keypad, to collect your cash."
    },
    "deposit": {
        "welcome": "Action: Welcome. To find the card slot, feel the area to the right of the physical keypad below the screen. Insert your card.",
        "enter pin": "Action: Move your hand below the screen to the keypad. Find the bump on the number 5, type your PIN, and press the green enter button.",
        "deposit": "Touch",
        "savings": "Touch",
        "insert cash": "Action: The cash deposit shutter is open. It is located below the keypad. Please insert your notes.",
        "confirm": "Touch"
    },
    "balance": {
        "welcome": "Action: Welcome. Please locate the card slot to the right of the physical keypad and insert your card.",
        "enter pin": "Action: Move your hand below the screen to the keypad. Find the bump on the number 5, type your PIN, and press the green enter button.",
        "balance": "Touch",
        "savings": "Touch",
        "print": "Touch"
    }
}

# ---------------------------
# INITIALIZATION
# ---------------------------
app = Flask(__name__)
load_dotenv()

endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
key = os.environ.get("AZURE_VISION_KEY")

if not endpoint or not key:
    raise ValueError("Missing Azure keys! Check your .env file.")

client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
print("[INFO] Loading YOLOv8 Pose Model...")
pose_model = YOLO("yolov8n-pose.pt")

# Dynamic State Variables
current_state = "READING_SCREEN"
dynamic_menu = {} # Stores { "1": [x1, y1, x2, y2], "2": [...] }
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
                for word in line.words:
                    pts = word.bounding_polygon
                    xs = [p.x for p in pts]
                    ys = [p.y for p in pts]
                    boxes[word.text.strip().lower()] = [min(xs), min(ys), max(xs), max(ys)]
    except Exception as e:
        print(f"[AZURE ERROR]: {e}")
    return boxes

def get_guidance_string(target_box, hand_x, hand_y):
    tx1, ty1, tx2, ty2 = target_box
    target_cx = (tx1 + tx2) // 2
    
    # FIX 1: The Wrist Offset. Aim the wrist 120 pixels BELOW the word, 
    # so the fingertips naturally land directly on the word.
    target_cy = ((ty1 + ty2) // 2) + 120 

    dx = target_cx - hand_x
    dy = target_cy - hand_y
    
    # FIX 2: Forgiving Hitbox. Increased from 60 to 100 to account for shaky hands.
    threshold = 100 

    if abs(dx) < threshold and abs(dy) < threshold: 
        return "You are on the button. Tap now."
    
    # FIX 3: Anti-Oscillation. Correct the axis that is furthest away first.
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
    
    requested_target = request.form.get('target') # Voice input from user
    
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    instruction = ""

    # ==========================================
    # STATE 1: READ THE SCREEN & NUMBER OPTIONS
    # ==========================================
    if current_state == "READING_SCREEN":
        boxes = get_text_bounding_boxes_from_azure(frame)
        screen_text_combined = " ".join(boxes.keys())
        
        # Check for static instructional screens first (Physical hardware tasks)
        if "welcome" in screen_text_combined or "insert card" in screen_text_combined:
            instruction = "Welcome. To find the card slot, feel the area to the right of the physical keypad below the screen. Insert your card."
        elif "enter pin" in screen_text_combined:
            instruction = "Screen asks for PIN. Move your hand directly below the center of the screen to find the physical keypad. Find the raised bump on the number 5 to orient your hand, type your PIN, and press the green enter button."
        elif "take cash" in screen_text_combined:
            instruction = "Transaction complete. Move your hand to the bottom of the ATM machine, far below the keypad, to collect your cash."
        elif "insert cash" in screen_text_combined:
            instruction = "The cash deposit shutter is open. It is located below the keypad. Please insert your notes."
        else:
            # We are on a menu! Let's dynamically number the buttons.
            dynamic_menu.clear()
            audio_options = "Options are: "
            counter = 1
            
            # Sort text from top to bottom of the screen
            sorted_texts = sorted(boxes.keys(), key=lambda k: boxes[k][1])
            
            for text in sorted_texts:
                box = boxes[text]
                # Calculate how tall the text is (y_max - y_min)
                box_height = box[3] - box[1] 
                
                # NOISE FILTER: Ignore tiny text (like stickers/tabs), 
                # ignore single random letters, and ignore long paragraphs.
                if len(text.split()) <= 4 and len(text) > 2 and box_height > 20: 
                    dynamic_menu[str(counter)] = box
                    audio_options += f"{counter} for {text}. "
                    counter += 1
            
            if counter > 1:
                instruction = audio_options + "Tap anywhere on the screen and say your number."
                current_state = "AWAITING_CHOICE"
                print(f"[INFO] Generated Menu: {dynamic_menu.keys()}")
            else:
                instruction = "Scanning screen..."

    # ==========================================
    # STATE 2: WAIT FOR USER TO SAY A NUMBER
    # ==========================================
    elif current_state == "AWAITING_CHOICE":
        if requested_target:
            req = requested_target.lower()
            matched_num = None
            
            # Map spoken words to integers
            word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5"}
            for word, num in word_to_num.items():
                if word in req:
                    matched_num = num
                    break
            
            if matched_num and matched_num in dynamic_menu:
                target_box = dynamic_menu[matched_num]
                instruction = f"Option {matched_num} selected. Show me your hand."
                current_state = "GUIDING_HAND"
            else:
                instruction = "I didn't catch a valid number. Tap the screen to try again."
        else:
            # Stay quiet while waiting for the frontend to send the voice command
            instruction = ""

    # ==========================================
    # STATE 3: GUIDE HAND & AUTO-RESET
    # ==========================================
    elif current_state == "GUIDING_HAND" and target_box:
        results = pose_model(frame, verbose=False)
        hand_x, hand_y = 0, 0
        
        for r in results:
            if r.keypoints and len(r.keypoints.xy) > 0:
                kpts = r.keypoints.xy[0] 
                if len(kpts) >= 11:
                    lx, ly = int(kpts[9][0]), int(kpts[9][1])
                    rx, ry = int(kpts[10][0]), int(kpts[10][1])
                    if rx > 0 and ry > 0: hand_x, hand_y = rx, ry
                    elif lx > 0 and ly > 0: hand_x, hand_y = lx, ly

        if hand_x > 0:
            instruction = get_guidance_string(target_box, hand_x, hand_y) 
            if instruction == "You are on the button. Tap now.":
                instruction = "Tap now. Then point the camera at the screen to read the next menu."
                current_state = "READING_SCREEN" # Auto-reset to read the new screen!
                target_box = None
        else:
            instruction = "Hand not visible. Please show your hand."

    return jsonify({"instruction": instruction, "state": current_state})

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"\n==========\nOPEN THIS URL ON YOUR PHONE:\n{public_url}\n==========\n")
    app.run(host='0.0.0.0', port=5000)
