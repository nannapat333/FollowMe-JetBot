import time
import cv2
import numpy as np
import importlib.util
import sys
import types
import speech_recognition as sr


# =========================
#  CONFIG
# =========================

FRAME_WIDTH  = 224
FRAME_HEIGHT = 224

# Distance estimation constant (similar to your Pikachu code)
K_DISTANCE = 3000.0            # tune this for your camera/object
STOP_DISTANCE_CM = 20.0        # stop when closer than this (approx)

# Set this to your mic index (use list_mics)
MIC_INDEX = 11                 # e.g. UACDemo / ReSpeaker

FORWARD_SPEED = 0.15            # how fast to drive forward
TURN_SPEED    = 0.1          # how fast to turn

# COLOR TO FOLLOW (HSV range)
# Right now: purple-ish like your doll
HSV_LOWER = np.array([80, 80, 50], dtype=np.uint8)
HSV_UPPER = np.array([100, 255, 255], dtype=np.uint8)
# For other colors, change these two lines only.


# =========================
#  Load JetBot modules dynamically
# =========================

jetbot_pkg = types.ModuleType("jetbot")
sys.modules["jetbot"] = jetbot_pkg

# --- jetbot.motor ---
spec_motor = importlib.util.spec_from_file_location(
    "jetbot.motor",
    "/usr/local/lib/python3.6/dist-packages/jetbot-0.4.3-py3.6.egg/jetbot/motor.py"
)
motor_module = importlib.util.module_from_spec(spec_motor)
sys.modules["jetbot.motor"] = motor_module
spec_motor.loader.exec_module(motor_module)

# --- jetbot.robot ---
spec_robot = importlib.util.spec_from_file_location(
    "jetbot.robot",
    "/usr/local/lib/python3.6/dist-packages/jetbot-0.4.3-py3.6.egg/jetbot/robot.py"
)
robot_module = importlib.util.module_from_spec(spec_robot)
sys.modules["jetbot.robot"] = robot_module
spec_robot.loader.exec_module(robot_module)

# --- jetbot.camera ---
spec_camera = importlib.util.spec_from_file_location(
    "jetbot.camera",
    "/usr/local/lib/python3.6/dist-packages/jetbot-0.4.3-py3.6.egg/jetbot/camera/__init__.py"
)
camera_module = importlib.util.module_from_spec(spec_camera)
sys.modules["jetbot.camera"] = camera_module
spec_camera.loader.exec_module(camera_module)

Robot  = robot_module.Robot
Camera = camera_module.Camera

robot  = Robot()
camera = Camera.instance(width=FRAME_WIDTH, height=FRAME_HEIGHT)

# Speech recognition objects (reuse them)
recognizer = sr.Recognizer()
microphone = sr.Microphone(device_index=MIC_INDEX)


# =========================
#  COLOR DETECTION + DISTANCE
# =========================

def detect_color_bbox(frame_bgr):
    """
    Find the largest region with the target color.
    Returns bounding box (x, y, w, h) or None.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Mask pixels inside [HSV_LOWER, HSV_UPPER]
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask   = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    # Ignore tiny blobs
    if area < 80:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h)


def estimate_distance_cm(bbox):
    """
    Approximate distance from bounding box height.
    Bigger h → closer. Using simple K_DISTANCE / h.
    """
    if bbox is None:
        return None
    x, y, w, h = bbox
    if h <= 0:
        return None
    return K_DISTANCE / float(h)


# =========================
#  ALIGN OBJECT TO IMAGE CENTER
# =========================

def align_color_center(robot, camera, tolerance_px=20, max_time=3.0):
    """
    Rotate left/right so that the color blob (bbox center)
    is near the horizontal center of the image.
    This is your 'set center of the object' step.
    """
    print(" Aligning color blob to image center...")
    TURN_SPEED_ALIGN = 0.1
    PULSE_TIME = 0.05

    start_time = time.time()

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_time:
                print("⏱ Alignment timeout, using best effort.")
                break

            frame_bgr = camera.value
            bbox = detect_color_bbox(frame_bgr)
            if bbox is None:
                print(" No color blob during alignment.")
                robot.stop()
                break

            x, y, w, h = bbox
            cx = x + w / 2.0

            # Error from image center
            error = cx - (FRAME_WIDTH / 2.0)
            print(f"Center x = {cx:.1f}, error = {error:.1f} px")

            if abs(error) <= tolerance_px:
                print(" Color blob centered within tolerance.")
                robot.stop()
                break

            if error > 0:
                print("↪️ Blob on right → turning right a bit")
                robot.right(TURN_SPEED_ALIGN)
            else:
                print("↩️ Blob on left → turning left a bit")
                robot.left(TURN_SPEED_ALIGN)

            time.sleep(PULSE_TIME)
            robot.stop()

    finally:
        robot.stop()
        print(" Finished alignment step.")


# =========================
#  MOVEMENT / FOLLOW LOGIC
# =========================

def follow_color_step():
    """
    One control step:
      - Grab frame
      - Detect color
      - If no color → spin to search
      - Estimate distance
      - Turn/drive based on position
      - STOP when too close (short range)
    """
    frame_bgr = camera.value
    bbox = detect_color_bbox(frame_bgr)

    # ===== SEARCH SPIN WHEN NO COLOR =====
    if bbox is None:
        print(" No target color detected → starting search spin.")
        SEARCH_TIME = 5.0  # seconds for one search attempt
        start = time.time()

        robot.left(TURN_SPEED)  # spin left slowly
        while time.time() - start < SEARCH_TIME:
            frame_bgr = camera.value
            bbox = detect_color_bbox(frame_bgr)
            if bbox is not None:
                print(" Found target color during search.")
                robot.stop()
                break
            time.sleep(0.05)
        else:
            # while ended without break → still nothing
            robot.stop()
            print(" Still no target color after search → stopping.")
            return

    # ===== from here on, bbox is NOT None =====
    x, y, w, h = bbox
    cx = x + w / 2.0
    center = FRAME_WIDTH / 2.0
    error  = cx - center

    # Estimate distance
    dist_cm = estimate_distance_cm(bbox)
    if dist_cm is not None:
        print(f" Estimated distance: {dist_cm:.1f} cm")
        # stop in short range so we don't hit the object
        if dist_cm <= STOP_DISTANCE_CM:
            print(f" Close enough (<= {STOP_DISTANCE_CM} cm). Stopping.")
            robot.stop()
            return

    CENTER_TOL = 20   # pixels; inside this = "centered enough"
    print(f" color cx={cx:.1f}, error={error:.1f}")

    # If color is roughly in the center → go forward
    if abs(error) <= CENTER_TOL:
        robot.left_motor.value  = FORWARD_SPEED
        robot.right_motor.value = FORWARD_SPEED

    # Color is on the right → turn right
    elif error > 0:
        robot.left_motor.value  = FORWARD_SPEED
        robot.right_motor.value = 0.0

    # Color is on the left → turn left
    else:
        robot.left_motor.value  = 0.0
        robot.right_motor.value = FORWARD_SPEED


# =========================
#  VOICE HELPERS
# =========================

def listen_once(timeout=5, phrase_time_limit=3):
    """
    Listen once and return recognized lowercased text,
    or '' if nothing/failed.
    """
    with microphone as source:
        print(" Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.7)
        try:
            audio = recognizer.listen(source,
                                      timeout=timeout,
                                      phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print(" No speech detected.")
            return ""

    try:
        text = recognizer.recognize_google(audio).lower()
        print(f" Heard: {text}")
        return text
    except sr.UnknownValueError:
        print(" Could not understand audio.")
    except sr.RequestError as e:
        print(f" Recognition error: {e}")
    return ""


def wait_for_go():
    """
    Stay in idle mode until user says 'follow me'.
    """
    print("🚦 Say 'follow me' to start color-follow mode.")
    while True:
        text = listen_once()
        if "follow me" in text:
            print(" 'follow me' detected → entering follow mode.")
            return


def check_for_stop_nonblocking():
    """
    Small/quick listen to see if user said 'stop' or 'start'.
    Returns True if we should exit follow mode.
    """
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        try:
            audio = recognizer.listen(source, timeout=0.5,
                                      phrase_time_limit=1.5)
        except sr.WaitTimeoutError:
            return False

    try:
        text = recognizer.recognize_google(audio).lower()
        print(f" (stop-check) Heard: {text}")
        # Accept both 'stop' and 'start' as exit triggers
        if "stop" in text or "start" in text:
            return True
    except (sr.UnknownValueError, sr.RequestError):
        pass

    return False


# =========================
#  MAIN BEHAVIOR
# =========================

def follow_color_until_stop():
    """
    Align first, then follow the color in a loop,
    until 'stop' (or 'start') is heard.
    """
    # First, align the blob to the center once
    align_color_center(robot, camera, tolerance_px=20, max_time=3.0)

    print(" Follow mode active. Say 'stop' to stop.")
    try:
        while True:
            # Do one movement step
            follow_color_step()

            # Brief pause so we don't hammer the CPU
            time.sleep(0.05)

            # Non-blocking check for "stop" / "start"
            if check_for_stop_nonblocking():
                print(" Voice command received → exiting follow mode.")
                robot.stop()
                break
    finally:
        # Always stop motors when leaving this function
        robot.stop()
        print("🔚 follow_color_until_stop finished.")


if __name__ == "__main__":
    try:
        # Idle until user says "follow me"
        while True:
            wait_for_go()              # blocks until "follow me"
            follow_color_until_stop()  # runs until "stop"

            print(" Back to idle. Say 'follow me' again to restart.")
    except KeyboardInterrupt:
        print("\n KeyboardInterrupt → cleaning up.")
    finally:
        robot.stop()
        try:
            camera.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Shutdown complete.")
