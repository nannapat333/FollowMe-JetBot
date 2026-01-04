Follow-Me JetBot
======================================

Project Overview
----------------
This project implements a JetBot robot that can:

1. Listen for the voice command "follow me".
2. Detect and track a target object based on its color (HSV range), blue in this case.
3. Drive towards the object while keeping it centered in the camera image.
4. Stop when it hears the voice command "stop" or when it is too close.

It combines classical computer vision for color tracking with
speech recognition for voice commands, running on a NVIDIA Jetson Nano
with the JetBot platform and a ReSpeaker USB microphone.


Key Features
------------
- Color-based target detection using HSV thresholding.
- Automatic alignment: JetBot rotates until the color blob is centered.
- Follow mode: JetBot moves forward and adjusts its direction to follow
  the color blob.
- Voice control:
  - Say "follow me" to start following.
  - Say "stop" (or "start", depending on code) to stop following.
- Safety stop when the robot is closer than a configurable distance.


Installation & Setup
--------------------
1. Copy the project to your JetBot:

2. Make sure you are using the JetBot environment:

3. Install required Python packages (if missing):
   - `pip install -r requirements.txt`

4. Check microphone index:
   - On a PC you can list devices with `python -m speech_recognition`.
   - In the script, set `MIC_INDEX = <your_index>` to match your USB mic.
   - On JetBot, you might need to try a few indices until it works.


How to Run
----------
1. Ensure the robot is on the floor with enough space to move.
2. Power on JetBot and connect via SSH or Jupyter terminal.
3. In a terminal:
(inside the project folder, run)
   - `python3 src.py`

4. Once started:
   - The script will initialize the camera, robot, and microphone.
   - It will print: "Say 'follow me' to start color-follow mode."

5. Usage:
   - Hold the target object with the blue color in front of
     the camera.
   - Say: **"follow me"**
   - The robot will:
     - Align the color blob to the image center.
     - Move forward and turn to keep the blob centered.
   - Say: **"stop"**  
     The robot stops and returns to idle mode.
   - You can say "follow me" again to restart.


Changing the Target Color
-------------------------
In `src.py` there are two key lines:

```python
HSV_LOWER = np.array([80, 80, 50], dtype=np.uint8)
HSV_UPPER = np.array([100, 255, 255], dtype=np.uint8)