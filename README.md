# AccessATM 🏧👁️
An AI-powered, voice-activated, and computer-vision-guided ATM assistant for visually impaired users. Built in 24 hours.

## The Problem
Modern ATMs rely entirely on visual digital screens, creating a massive barrier for visually impaired users. Standard screen readers cannot bridge the gap between digital interfaces and physical hardware components (like the card slot or physical keypad).

## The Solution
AccessATM acts as a continuous, multimodal AI agent running locally on a smartphone. 
* **Real-Time OCR & Spatial Cropping:** Uses Azure AI Vision to constantly scan the ATM screen, employing a 75/25 mathematical spatial crop to ignore physical stickers and only read digital menus.
* **Target Lock Hand Guidance:** Uses MediaPipe's Modern Tasks API to track the user's index fingertip. It mathematically guides the user's hand (X/Y axes) to the correct button and explicitly commands the Z-axis tap.
* **Global Intent Voice Recognition:** Users can navigate, ask for context, or cancel actions using natural voice commands.

## How to Run the Solution Locally

### Prerequisites
* Python 3.12+
* An Azure AI Vision Endpoint and API Key
* A smartphone for the frontend camera/microphone feed

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ndumisoskhosana/AccessATM.git
   cd AccessATM
