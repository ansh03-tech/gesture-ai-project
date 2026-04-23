"""
collector.py — Hand Landmark Extraction + Data Collection
"""

import cv2
import numpy as np
import mediapipe as mp
import csv
import os


def extract_landmarks(frame: np.ndarray, hands_detector) -> list | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]

    raw = [(lm.x, lm.y) for lm in hand.landmark]

    wrist_x, wrist_y = raw[0]
    centered = [(x - wrist_x, y - wrist_y) for x, y in raw]

    flat = [v for pair in centered for v in pair]
    max_val = max(abs(v) for v in flat) or 1.0
    normalised = [v / max_val for v in flat]

    return normalised


# ─────────────────────────────────────────────
# ✅ DATA COLLECTION RUNNER (ADDED SAFELY)
# ─────────────────────────────────────────────

def collect_data(label: str, save_path="gestures_data.csv"):
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access camera")
        return

    print(f"📸 Collecting data for gesture: {label}")
    print("👉 Press 's' to save sample")
    print("👉 Press 'q' to quit")

    with mp_hands.Hands(max_num_hands=1) as hands:
        with open(save_path, "a", newline="") as f:
            writer = csv.writer(f)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                landmarks = extract_landmarks(frame, hands)

                cv2.putText(frame, f"Gesture: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Collector", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('s') and landmarks:
                    writer.writerow(landmarks + [label])
                    print("✅ Sample saved")

                elif key == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("📁 Data collection finished")


# ─────────────────────────────────────────────
# ✅ ENTRY POINT (VERY IMPORTANT)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    label = input("Enter gesture label: ")
    collect_data(label)