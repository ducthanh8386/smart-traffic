import numpy as np
import cv2
from core.traffic_light_detector import TrafficLightDetector


def test_traffic_light_detector_red_mask():
    detector = TrafficLightDetector()
    # Create a synthetic image with a bright red circle
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (0, 0, 255), -1)  # Pure red in BGR

    state = detector.detect_state(img)
    assert state == "RED"


def test_traffic_light_detector_green_mask():
    detector = TrafficLightDetector()
    # Create a synthetic image with a bright green circle
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (0, 255, 0), -1)  # Pure green in BGR

    state = detector.detect_state(img)
    assert state == "GREEN"


def test_traffic_light_detector_unknown_on_black():
    detector = TrafficLightDetector()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    state = detector.detect_state(img)
    assert state == "UNKNOWN"
