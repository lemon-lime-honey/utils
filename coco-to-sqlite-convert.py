import sqlite3
import ijson
import json
from decimal import Decimal
from pathlib import Path
from tqdm import tqdm


BATCH_SIZE = 10000


def create_tables(cursor):
    cursor.execute("DROP TABLE IF EXISTS images")
    cursor.execute("DROP TABLE IF EXISTS annotations")
    cursor.execute("DROP TABLE IF EXISTS categories")

    cursor.execute(
        """
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            file_name TEXT,
            width INTEGER,
            height INTEGER,
            license INTEGER,
            flickr_url TEXT,
            coco_url TEXT,
            date_captured TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE annotations (
            id INTEGER PRIMARY KEY,
            image_id INTEGER,
            category_id INTEGER,
            bbox TEXT,
            area REAL,
            iscrowd INTEGER,
            segmentation TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name TEXT,
            supercategory TEXT
        )
        """
    )


def insert_batch(cursor, sql, batch):
    cursor.executemany(sql, batch)


def stream_categories(json_path, cursor, conn):
    batch = []
    with open(json_path, "rb") as f:
        for cat in ijson.items(f, "categories.item"):
            batch.append(
                (
                    cat["id"],
                    cat["name"].strip(),
                    cat.get("supercategory", ""),
                )
            )

            if len(batch) >= BATCH_SIZE:
                insert_batch(
                    cursor,
                    "INSERT INTO categories (id, name, supercategory) VALUES (?, ?, ?)",
                    batch,
                )
                conn.commit()
                batch.clear()

    if batch:
        insert_batch(
            cursor,
            "INSERT INTO categories (id, name, supercategory) VALUES (?, ?, ?)",
            batch,
        )
        conn.commit()


def stream_images(json_path, cursor, conn):
    batch = []
    with open(json_path, "rb") as f:
        for img in tqdm(ijson.items(f, "images.item"), desc="Images"):
            batch.append(
                (
                    img["id"],
                    img["file_name"],
                    img["width"],
                    img["height"],
                    img.get("license"),
                    img.get("flickr_url"),
                    img.get("coco_url"),
                    img.get("date_captured"),
                )
            )

            if len(batch) >= BATCH_SIZE:
                insert_batch(
                    cursor,
                    """
                    INSERT INTO images 
                    (id, file_name, width, height, license, flickr_url, coco_url, date_captured)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                batch.clear()

    if batch:
        insert_batch(
            cursor,
            """
            INSERT INTO images 
            (id, file_name, width, height, license, flickr_url, coco_url, date_captured)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()


def stream_annotations(json_path, cursor, conn):
    batch = []
    with open(json_path, "rb") as f:
        for ann in tqdm(ijson.items(f, "annotations.item"), desc="Annotations"):
            bbox = normalize_json_numbers(ann.get("bbox", []))
            segmentation = normalize_json_numbers(ann.get("segmentation", []))

            area = ann.get("area", 0.0)
            iscrowd = ann.get("iscrowd", 0)

            batch.append(
                (
                    int(ann["id"]),
                    int(ann["image_id"]),
                    int(ann["category_id"]),
                    json.dumps(bbox),
                    float(area),
                    int(iscrowd),
                    json.dumps(segmentation),
                )
            )

            if len(batch) >= BATCH_SIZE:
                insert_batch(
                    cursor,
                    """
                    INSERT INTO annotations 
                    (id, image_id, category_id, bbox, area, iscrowd, segmentation)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                batch.clear()

    if batch:
        insert_batch(
            cursor,
            """
            INSERT INTO annotations 
            (id, image_id, category_id, bbox, area, iscrowd, segmentation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()


def create_indexes(cursor, conn):
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_annotations_image_id
        ON annotations(image_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_annotations_category_id
        ON annotations(category_id)
    """)
    conn.commit()


def normalize_json_numbers(obj):
    """
    Recursively convert Decimal to float for JSON serialization.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, list):
        return [normalize_json_numbers(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: normalize_json_numbers(v) for k, v in obj.items()}
    else:
        return obj


def build_coco_database_streaming(json_path: Path, db_path: Path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_tables(cursor)
    conn.commit()

    print("[1/3] Streaming categories...")
    stream_categories(json_path, cursor, conn)

    print("[2/3] Streaming images...")
    stream_images(json_path, cursor, conn)

    print("[3/3] Streaming annotations...")
    stream_annotations(json_path, cursor, conn)

    print("Creating indexes...")
    create_indexes(cursor, conn)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    json_path = Path("target.json")
    db_path = Path("db.sqlite3")

    build_coco_database_streaming(json_path, db_path)
