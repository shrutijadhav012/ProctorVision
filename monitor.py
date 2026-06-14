import cv2
import numpy as np
import speech_recognition as sr
import threading
import time
import datetime
import urllib.request
import os
from collections import deque


from ultralytics import YOLO
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.columns import Columns
from rich import box


import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    RunningMode,
    FaceLandmarksConnections,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision import drawing_utils as mp_draw
from mediapipe.tasks.python.vision import drawing_styles as mp_styles_mod


CONFIG = {
    "CAMERA_INDEX":        0,
    "FRAME_WIDTH":         1280,
    "FRAME_HEIGHT":        720,
    "MULTI_PERSON_THRESH": 1,
    "HEAD_TILT_THRESH":    25,   # degrees
    "HEAD_TURN_THRESH":    30,   # degrees
    "GADGET_CLASSES": {
        67: "Phone",
        63: "Laptop",
        62: "TV",
        65: "Book",
        66: "Mouse",
        64: "Monitor",
        76: "Headphones",
    },
    "AUDIO_ENABLED":    True,
    "LOG_FILE":         "proctor_log.txt",
    "WARNING_COOLDOWN": 3,
    "MAX_LOG_LINES":    10,
    
    "FACE_MODEL_URL": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "HAND_MODEL_URL": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "FACE_MODEL_PATH": "face_landmarker.task",
    "HAND_MODEL_PATH": "hand_landmarker.task",
}

console = Console()



def download_model(url: str, path: str, label: str):
    if os.path.exists(path):
        console.print(f"  [green]OK[/green] {label} already cached")
        return
    console.print(f"  [yellow]...[/yellow] Downloading {label}...")
    urllib.request.urlretrieve(url, path)
    console.print(f"  [green]OK[/green] {label} downloaded")



class WarningManager:
    def __init__(self):
        self.warnings       = deque(maxlen=CONFIG["MAX_LOG_LINES"])
        self.last_triggered = {}
        self.counts         = {}

    def trigger(self, key: str, message: str, level: str = "WARN") -> bool:
        now = time.time()
        if now - self.last_triggered.get(key, 0) < CONFIG["WARNING_COOLDOWN"]:
            return False
        self.last_triggered[key] = now
        self.counts[key] = self.counts.get(key, 0) + 1
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.warnings.appendleft({
            "time": ts, "level": level,
            "message": message, "count": self.counts[key],
        })
        with open(CONFIG["LOG_FILE"], "a") as f:
            f.write(f"[{ts}] [{level}] {message}\n")
        return True

    def get_table(self) -> Table:
        tbl = Table(
            box=box.SIMPLE_HEAD, show_header=True,
            header_style="bold cyan", expand=True,
            border_style="bright_black",
        )
        tbl.add_column("Time",  style="dim",   width=10)
        tbl.add_column("Level", style="bold",  width=10)
        tbl.add_column("Event", style="white", ratio=1)
        tbl.add_column("x",     justify="right", width=4)

        colours = {"WARN": "yellow", "ALERT": "red", "INFO": "green"}
        icons   = {"WARN": "!",      "ALERT": "!!",  "INFO": "i"}
        for w in list(self.warnings):
            c = colours.get(w["level"], "white")
            i = icons.get(w["level"], "*")
            tbl.add_row(
                w["time"],
                f"[{c}]{i} {w['level']}[/{c}]",
                w["message"],
                f"[{c}]{w['count']}[/{c}]",
            )
        return tbl



class AudioTranscriber:
    def __init__(self):
        self.transcript = deque(maxlen=6)
        self.running    = False
        self._thread    = None
        self.r          = sr.Recognizer()
        self.r.energy_threshold         = 300
        self.r.dynamic_energy_threshold = True

    def _listen_loop(self):
        try:
            with sr.Microphone() as mic:
                self.r.adjust_for_ambient_noise(mic, duration=1)
                while self.running:
                    try:
                        audio = self.r.listen(mic, timeout=3, phrase_time_limit=5)
                        text  = self.r.recognize_google(audio)
                        ts    = datetime.datetime.now().strftime("%H:%M:%S")
                        self.transcript.appendleft(f"[dim]{ts}[/dim]  {text}")
                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        pass
                    except Exception:
                        pass
        except Exception as e:
            self.transcript.appendleft(f"[red]Mic error: {e}[/red]")

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def get_panel(self) -> Panel:
        lines = list(self.transcript) or ["[dim]Listening for speech...[/dim]"]
        return Panel(
            "\n".join(lines),
            title="[bold cyan]  Audio Transcript[/bold cyan]",
            border_style="cyan", padding=(0, 1),
        )



class StatusTracker:
    def __init__(self):
        self.persons     = 0
        self.head_pose   = "-"
        self.hands       = "-"
        self.gadgets     = []
        self.fps         = 0.0
        self.frame_count = 0
        self._fps_times  = deque(maxlen=30)

    def tick(self):
        self._fps_times.append(time.time())
        self.frame_count += 1
        if len(self._fps_times) >= 2:
            self.fps = (len(self._fps_times) - 1) / (
                self._fps_times[-1] - self._fps_times[0] + 1e-9)

    def get_status_panel(self) -> Panel:
        tbl = Table(
            box=box.SIMPLE, show_header=False,
            expand=True, border_style="bright_black",
        )
        tbl.add_column("Key",   style="dim",   width=16)
        tbl.add_column("Value", style="white", ratio=1)

        ps = "red bold" if self.persons > CONFIG["MULTI_PERSON_THRESH"] else "green bold"
        tbl.add_row("Persons",   f"[{ps}]{self.persons}[/{ps}]")
        tbl.add_row("Head Pose", self.head_pose)
        tbl.add_row("Hands",     self.hands)
        tbl.add_row("Gadgets",
            "  ".join(self.gadgets) if self.gadgets else "[dim]None[/dim]")
        tbl.add_row("FPS",       f"[cyan]{self.fps:.1f}[/cyan]")
        tbl.add_row("Frame",     f"[dim]{self.frame_count}[/dim]")
        return Panel(
            tbl, title="[bold]  Live Status[/bold]",
            border_style="bright_blue", padding=(0, 1),
        )



class ProctoringSystem:
    def __init__(self):
        console.print(Panel.fit(
            "[bold cyan]Initialising Proctor AI...[/bold cyan]",
            border_style="cyan"))

        
        download_model(CONFIG["FACE_MODEL_URL"],
                       CONFIG["FACE_MODEL_PATH"], "Face Landmarker model")
        download_model(CONFIG["HAND_MODEL_URL"],
                       CONFIG["HAND_MODEL_PATH"], "Hand Landmarker model")

        
        face_opts = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=CONFIG["FACE_MODEL_PATH"]),
            running_mode=RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face = FaceLandmarker.create_from_options(face_opts)
        console.print("  [green]OK[/green] Face Landmarker ready")

        
        hand_opts = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=CONFIG["HAND_MODEL_PATH"]),
            running_mode=RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands = HandLandmarker.create_from_options(hand_opts)
        console.print("  [green]OK[/green] Hand Landmarker ready")

        
        self.yolo = YOLO("yolov8n.pt")
        console.print("  [green]OK[/green] YOLOv8 loaded")

        
        self.warnings = WarningManager()
        self.status   = StatusTracker()
        self.audio    = AudioTranscriber()

        if CONFIG["AUDIO_ENABLED"]:
            self.audio.start()
            console.print("  [green]OK[/green] Audio transcriber started")

        
        self.cap = cv2.VideoCapture(CONFIG["CAMERA_INDEX"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["FRAME_WIDTH"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
        if not self.cap.isOpened():
            raise RuntimeError(
                "Camera not found. Change CAMERA_INDEX in CONFIG.")
        console.print("  [green]OK[/green] Camera opened\n")

    
    def _head_pose(self, landmarks, w, h):
        def pt(i):
            return np.array([landmarks[i].x * w, landmarks[i].y * h])
        nose, chin = pt(1), pt(152)
        leye, reye = pt(33), pt(263)
        tilt = np.degrees(np.arctan2(
            (chin - nose)[0], (chin - nose)[1]))
        turn = np.degrees(np.arctan2(
            (reye - leye)[1], (reye - leye)[0]))
        return tilt, turn

    
    def _draw_face(self, frame, face_landmarks_list):
        h, w = frame.shape[:2]
        connections = [
            (c.start, c.end)
            for c in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
        ]
        for lm_list in face_landmarks_list:
            pts = [(int(l.x * w), int(l.y * h)) for l in lm_list]
            for s, e in connections:
                if s < len(pts) and e < len(pts):
                    cv2.line(frame, pts[s], pts[e], (0, 200, 180), 1)

    
    def _draw_hands(self, frame, hand_landmarks_list):
        h, w = frame.shape[:2]
        connections = [
            (c.start, c.end)
            for c in HandLandmarksConnections.HAND_CONNECTIONS
        ]
        for lm_list in hand_landmarks_list:
            pts = [(int(l.x * w), int(l.y * h)) for l in lm_list]
            for s, e in connections:
                cv2.line(frame, pts[s], pts[e], (0, 255, 100), 2)
            for px, py in pts:
                cv2.circle(frame, (px, py), 4, (255, 80, 0), -1)

    
    def _draw_overlay(self, frame, persons, num_hands, gadgets):
        h, w = frame.shape[:2]
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (w, 46), (12, 12, 22), -1)
        cv2.addWeighted(bar, 0.78, frame, 0.22, 0, frame)

        txt = (f"  PROCTOR AI"
               f"  |  Persons: {persons}"
               f"  |  Hands: {num_hands}"
               f"  |  Gadgets: {len(gadgets)}"
               f"  |  FPS: {self.status.fps:.1f}"
               f"  |  {datetime.datetime.now().strftime('%H:%M:%S')}")
        cv2.putText(frame, txt, (8, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 215, 255), 1, cv2.LINE_AA)

        if persons > CONFIG["MULTI_PERSON_THRESH"]:
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 180), -1)
            cv2.putText(
                frame,
                f"  WARNING: MULTIPLE PERSONS DETECTED ({persons} faces)",
                (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return frame

    
    def run(self):
        console.print(Panel(
            "[bold green]Camera active — press Q in the video window to quit[/bold green]",
            border_style="green"))

        with Live(self._build_dashboard(), refresh_per_second=4,
                  screen=False, console=console) as live:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.status.tick()
                h, w = frame.shape[:2]

                
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )

                
                face_result = self._face.detect(mp_image)
                num_faces   = 0
                head_labels = []

                if face_result.face_landmarks:
                    num_faces = len(face_result.face_landmarks)
                    self._draw_face(frame, face_result.face_landmarks)
                    for lm_list in face_result.face_landmarks:
                        tilt, turn = self._head_pose(lm_list, w, h)
                        label = "Forward"
                        if abs(tilt) > CONFIG["HEAD_TILT_THRESH"]:
                            label = f"Tilted {tilt:+.0f}deg"
                            self.warnings.trigger(
                                "head_tilt",
                                f"Head tilt detected ({tilt:+.0f}deg)", "WARN")
                        if abs(turn) > CONFIG["HEAD_TURN_THRESH"]:
                            label = f"Turned {turn:+.0f}deg"
                            self.warnings.trigger(
                                "head_turn",
                                f"Head turned away ({turn:+.0f}deg)", "WARN")
                        head_labels.append(label)

                if num_faces > CONFIG["MULTI_PERSON_THRESH"]:
                    self.warnings.trigger(
                        "multi_person",
                        f"Multiple persons — {num_faces} faces visible", "ALERT")

                
                hand_result = self._hands.detect(mp_image)
                hand_labels = []

                if hand_result.hand_landmarks:
                    self._draw_hands(frame, hand_result.hand_landmarks)
                    for i, hand_world in enumerate(hand_result.hand_landmarks):
                        side = "Right"
                        if hand_result.handedness and i < len(hand_result.handedness):
                            side = hand_result.handedness[i][0].display_name
                        hand_labels.append(side)
                    self.warnings.trigger(
                        "hand_visible",
                        f"{len(hand_labels)} hand(s): {', '.join(hand_labels)}", "INFO")

                
                yolo_res = self.yolo(frame, verbose=False)[0]
                gadgets  = []
                for bd in yolo_res.boxes:
                    cls_id = int(bd.cls[0])
                    if cls_id in CONFIG["GADGET_CLASSES"]:
                        conf  = float(bd.conf[0])
                        label = CONFIG["GADGET_CLASSES"][cls_id]
                        gadgets.append(label)
                        x1, y1, x2, y2 = map(int, bd.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 80, 255), 2)
                        cv2.rectangle(frame, (x1, y1 - 26), (x2, y1),
                                      (0, 80, 255), -1)
                        cv2.putText(frame, f"{label} {conf:.0%}",
                                    (x1 + 4, y1 - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                        self.warnings.trigger(
                            f"gadget_{cls_id}",
                            f"Gadget: {label} ({conf:.0%})", "ALERT")

               
                self.status.persons   = num_faces
                self.status.head_pose = (", ".join(head_labels)
                                         if head_labels else "-")
                self.status.hands     = (
                    f"{len(hand_labels)} ({', '.join(hand_labels)})"
                    if hand_labels else "None")
                self.status.gadgets   = list(set(gadgets))

                frame = self._draw_overlay(
                    frame, num_faces, len(hand_labels), gadgets)
                cv2.imshow("Proctor AI  |  press Q to quit", frame)
                live.update(self._build_dashboard())

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self._shutdown()

    def _build_dashboard(self):
        warn_panel = Panel(
            self.warnings.get_table(),
            title="[bold yellow]  Event Log[/bold yellow]",
            border_style="yellow", padding=(0, 1),
        )
        return Panel(
            Columns([
                self.status.get_status_panel(),
                Columns([warn_panel, self.audio.get_panel()], equal=True),
            ]),
            title="[bold white on blue]  PROCTOR AI  -  Intelligent Monitoring System  [/bold white on blue]",
            border_style="blue", padding=(0, 1),
        )

    def _shutdown(self):
        console.print("\n[yellow]Shutting down...[/yellow]")
        self.audio.stop()
        self._face.close()
        self._hands.close()
        self.cap.release()
        cv2.destroyAllWindows()
        console.print(Panel.fit(
            f"[bold green]Session ended. Log: [cyan]{CONFIG['LOG_FILE']}[/cyan][/bold green]",
            border_style="green"))


if __name__ == "__main__":
    try:
        system = ProctoringSystem()
        system.run()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
