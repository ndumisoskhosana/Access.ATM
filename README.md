# ATM Assistive Vision System

---

## 🎥 Demo Video
<p align="center">
<a href="YOUR_VIDEO_LINK_HERE" target="_blank"><b>▶ Watch the Demo Video</b></a>
</p>

---

## 📌 **Overview**

This project is an <u>AI-powered assistive system</u> designed to help visually impaired users successfully withdraw cash from ATM machines using <u>computer vision</u>, <u>voice guidance</u>, and <u>speech recognition</u>.

The system uses a live camera feed to:
- <u>Detect ATM components</u> (screen, keypad, card slot, cash dispenser)
- <u>Read on-screen text</u> using OCR (via Microsoft Azure)
- <u>Guide the user</u> with real-time audio instructions
- <u>Assist the user</u> in physically interacting with the ATM to complete a withdrawal

---

## 📌 **Problem**

Visually impaired users face significant barriers when using ATMs:
- Interfaces rely heavily on <u>visual cues</u>
- Button layouts vary between machines
- Cash slots and card slots are difficult to locate
- Existing accessibility features are limited or inconsistent

This makes <u>independent cash withdrawal</u> difficult or impossible in many real-world situations.

---

## 📌 **Solution**

This project is an <u>AI-assisted guidance system</u> that helps visually impaired users successfully withdraw cash from ATMs using:
- <u>Computer vision</u>
- <u>Voice guidance</u>
- <u>Speech recognition</u>

The system uses a camera to interpret the ATM environment and provides <u>real-time audio instructions</u> to guide the user step-by-step.

---

## 📌 **Impact**

**This project demonstrates how AI can:**
- <u>Improve financial accessibility</u>
- <u>Enable independent banking</u>
- <u>Reduce reliance</u> on assistance from others
- <u>Bridge the gap</u> between digital interfaces and physical interaction

---

## 📌 **Future Improvements**

- Expand functionality beyond cash withdrawal to support:
  - Cash deposits
  - PIN changes
  - Printing bank statements
- Train a custom ATM-specific detection model
- Add <u>multilingual voice support</u>
- Improve robustness in noisy environments
- Deploy as a <u>mobile application</u>

---

## 📚 **Libraries Used**

---

### 📊 Table Format
<table>
  <tr>
    <th>Library</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td><b>opencv-python (cv2)</b></td>
    <td>Captures video from webcam and processes image frames</td>
  </tr>
  <tr>
    <td><b>pyttsx3</b></td>
    <td>Converts text to speech for real-time audio guidance</td>
  </tr>
  <tr>
    <td><b>SpeechRecognition</b></td>
    <td>Captures and processes user voice input</td>
  </tr>
  <tr>
    <td><b>re</b></td>
    <td>Extracts information (e.g. Rand amounts) from OCR text</td>
  </tr>
  <tr>
    <td><b>time</b></td>
    <td>Handles timing logic (e.g. ATM visibility timeout)</td>
  </tr>
  <tr>
    <td><b>ultralytics (YOLOv8)</b></td>
    <td>Object detection for ATM components (screen, keypad, etc.)</td>
  </tr>
  <tr>
    <td><b>azure-ai-vision-imageanalysis</b></td>
    <td>Performs OCR to read ATM screen text</td>
  </tr>
  <tr>
    <td><b>azure-core</b></td>
    <td>Handles authentication and communication with Azure services</td>
  </tr>
</table>
</table>
