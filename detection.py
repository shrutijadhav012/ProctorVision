import cv2
import mediapipe as mp
from ultralytics import YOLO
import os
import time
import matplotlib
matplotlib.use('Agg')  

class AttentionMonitor:
    def __init__(self):
        
        self.yolo = YOLO("yolov8n.pt")

        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        
        self.ss_dir = "screenshots"
        os.makedirs(self.ss_dir, exist_ok=True)

    def detect_gadget(self, frame):
        """Use YOLOv8 to detect gadgets like phones, laptops, tablets, and other prohibited items."""
        results = self.yolo.predict(frame, verbose=False)
        gadgets = []
        prohibited_items = {
            "cell phone": "Mobile Phone",
            "laptop": "Laptop Computer", 
            "keyboard": "External Keyboard",
            "tv": "TV/Monitor",
            "remote": "Remote Control",
            "mouse": "Computer Mouse",
            "tablet": "Tablet Device",
            "book": "Book/Notes",
            "bottle": "Water Bottle", 
            "cup": "Cup/Mug",
            "smartphone": "Smartphone",
            "calculator": "Calculator",
            "headphones": "Headphones/Earphones",
            "microphone": "Microphone",
            "camera": "Camera Device",
            "watch": "Smart Watch",
            "glasses": "Smart Glasses"
        }
        
        if results and len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = results[0].names[cls_id]
                
                
                for item_key, item_display in prohibited_items.items():
                    if item_key in label.lower():
                        gadgets.append(item_display)
                        break
                
                
                if label in prohibited_items:
                    gadgets.append(prohibited_items[label])
        
        return list(set(gadgets))  

    def detect_head_turn(self, face_landmarks, w):
        """Check if person looking left/right/forward."""
        nose = face_landmarks.landmark[1]  
        nose_x = nose.x * w
        center_x = w // 2
        if nose_x < center_x - 60:
            return "Looking Left"
        elif nose_x > center_x + 60:
            return "Looking Right"
        else:
            return "Looking Forward"

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        face_results = self.face_mesh.process(rgb)
        head_status = "No Face Detected"
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                head_status = self.detect_head_turn(face_landmarks, w)

        
        hand_results = self.hands.process(rgb)
        hand_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0

        
        gadgets = self.detect_gadget(frame)

        
        warnings = []
        
        
        if head_status == "No Face Detected":
            warnings.append("⚠️ FACE NOT VISIBLE - Position yourself in camera view")
        elif head_status == "Looking Left":
            warnings.append("👈 HEAD TURNED LEFT - Look straight at the camera")
        elif head_status == "Looking Right":
            warnings.append("👉 HEAD TURNED RIGHT - Look straight at the camera")
        
        
        if hand_count == 0:
            warnings.append("✋ NO HANDS VISIBLE - Keep both hands on the desk")
        elif hand_count == 1:
            warnings.append("🖐️ ONLY ONE HAND VISIBLE - Show both hands on desk")
        
        if gadgets:
            for gadget in gadgets:
                warnings.append(f"📱 PROHIBITED DEVICE: {gadget} - Remove immediately")

        return warnings, head_status, hand_count, gadgets

    def save_screenshot(self, frame):
        """Save a screenshot when gadget is detected."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.ss_dir, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[INFO] Screenshot saved at {filename}")

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            warnings, head_status, hand_count, gadgets = self.process_frame(frame)

            
            if head_status == "Looking Forward" and hand_count == 2 and not gadgets:
                color = (0, 255, 0)  
            else:
                color = (0, 0, 255)  

            cv2.putText(frame, f"Head: {head_status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Hands: {hand_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            
            y_offset = 120
            for warning in warnings:
                cv2.putText(frame, warning, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y_offset += 40

            
            if gadgets:
                self.save_screenshot(frame)

            cv2.imshow("Attention Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor = AttentionMonitor()
    monitor.run()
