import sqlite3
from pathlib import Path
from ultralytics import YOLO

# --- [Settings] ---
VIDEO_DIR = Path("/path/to/videos")
DB_PATH = Path("/path/to/db.db")
MODEL_PATH = "/path/to/model.pt"
CONF_THRESHOLD = 0.25
ROI = [960, 0, 1920, 1080]  # [x1, y1, x2, y2]
TARGET_CLASSES = [0]
# ------------------


def init_db(path):
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                frame_no INTEGER,
                class_name TEXT,
                confidence REAL,
                x_center_rel REAL, y_center_rel REAL,
                width_rel REAL, height_rel REAL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            );
            CREATE INDEX IF NOT EXISTS idx_video_id ON detections(video_id);
        """)


def run():
    model = YOLO(MODEL_PATH)
    init_db(DB_PATH)

    names = model.names
    video_files = [
        f for f in VIDEO_DIR.iterdir() if f.suffix.lower() in (".mp4", ".avi", ".mov")
    ]

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        for v_path in video_files:
            print(f"시작: {v_path.name}")

            cursor.execute(
                "INSERT OR IGNORE INTO videos (filename) VALUES (?)", (v_path.name,)
            )
            cursor.execute("SELECT id FROM videos WHERE filename = ?", (v_path.name,))
            video_id = cursor.fetchone()[0]

            results = model.predict(
                source=str(v_path),
                stream=True,
                conf=CONF_THRESHOLD,
                classes=TARGET_CLASSES,
                verbose=False,
            )

            for frame_idx, res in enumerate(results):
                batch = []
                if res.boxes is not None:
                    for box in res.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                        if ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]:
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            xywhn = box.xywhn[0].tolist()

                            batch.append(
                                (video_id, frame_idx, names[cls], conf, *xywhn)
                            )

                if batch:
                    cursor.executemany(
                        """
                        INSERT INTO detections (video_id, frame_no, class_name, confidence, 
                        x_center_rel, y_center_rel, width_rel, height_rel) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        batch,
                    )

            conn.commit()
            print(f"저장 완료: {v_path.name}")


if __name__ == "__main__":
    run()
