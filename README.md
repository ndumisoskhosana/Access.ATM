# AccessATM 🏧👁️
**A Voice-Activated, Spatially-Aware AI Agent for Visually Impaired Users**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Azure AI Vision](https://img.shields.io/badge/Azure-AI%20Vision-0078D4.svg)](https://azure.microsoft.com/)
[![MediaPipe](https://img.shields.io/badge/Google-MediaPipe-EA4335.svg)](https://developers.google.com/mediapipe)
[![Flask](https://img.shields.io/badge/Backend-Flask-black.svg)](https://flask.palletsprojects.com/)

🔗 **[Watch the 4.5-Minute End-to-End Demo Video Here](https://lnkd.in/dYpjYXUV)**

## 📌 The Problem
Modern ATMs rely entirely on visual digital screens, creating a massive barrier for visually impaired users. While standard digital screen readers exist, they **cannot bridge the gap to physical hardware**. They cannot tell a user where the physical card slot is, orient their hand on a metal keypad, or differentiate between a digital button and a physical braille sticker. 

## 💡 The Solution
AccessATM is a continuous, multimodal AI agent running locally via a smartphone camera. It acts as an autonomous digital-to-physical bridge, reading digital menus dynamically while utilizing computer vision to guide the user's hand to physical touchpoints in the real world. 

## 🚀 Key Technical Engineering Achievements

This prototype moves beyond simple API wrapping by implementing robust, enterprise-grade safety nets and spatial logic:

* **The 75/25 Spatial Mathematical Crop:** Standard OCR models hallucinate when reading physical environments. I engineered a dynamic mathematical crop that completely ignores text detected in the bottom 25% of the camera frame. This prevents the AI from accidentally reading physical hardware stickers (like "Insert Card Here") and turning them into clickable digital menu buttons.
* **Continuous Reactive State Machine with Auto-Wakeup:** The backend does not wait for user prompts. It continuously scans the camera feed at 1.2-second intervals. By comparing the intersection of word sets between frames, the AI autonomously detects when the physical screen changes (e.g., jumping from a PIN screen to a Menu), automatically waking up to generate and read the new menu without requiring a voice command.
* **OCR Jitter Deduplication Filter:** To prevent "audio spam" caused by minor camera pixel shifts triggering false screen updates, I implemented a strict deduplication and static-context filter. It intelligently separates actionable buttons from static text (like account balances), reading the context once and seamlessly dropping into a silent, paused state to await user commands.
* **MediaPipe Target Lock Protocol:** Standard AI tracks the wrist, creating a dangerous accuracy gap for visually impaired users. I utilized MediaPipe's Modern Tasks API to track the specific index fingertip (Landmark 8). Because a 2D camera lacks Z-axis depth, I engineered a mathematical vector offset that guides the user on the X/Y axes and explicitly commands a forward Z-axis push *only* when the target lock threshold is achieved.
* **Global Intent Voice Engine:** Built an asynchronous, state-agnostic voice listener. Regardless of where the user is in the menu loop, they can globally interrupt the AI to ask for physical context ("Where is the keypad?"), request repetition, or trigger an emergency "Cancel" to instantly sever the hand-tracking loop and reset the system.

## 🛠️ Installation & Local Implementation

To run this prototype locally, you will need an Azure AI Vision Endpoint and a smartphone to act as the frontend camera/microphone.

### 1. Clone the Repository
```bash
git clone [https://github.com/ndumisoskhosana/Access.ATM.git](https://github.com/ndumisoskhosana/Access.ATM.git)
cd Access.ATM