import sqlite3
import json
import os
from pathlib import Path
from tqdm import tqdm


def load_category_mapping(cursor):
    mapping = {}
    cursor.execute("SELECT id, name FROM categories")
    for cat_id, name in cursor:
        mapping[name] = cat_id
    return mapping


def fetch_filtered_images(cursor, image_path: Path):
    target_images = set(os.listdir(image_path))
    images = []
    image_ids = []

    cursor.execute(
        "SELECT id, file_name, width, height, license, flickr_url, coco_url, date_captured FROM images"
    )

    for row in tqdm(cursor, desc="Filter DB Images"):
        (
            img_id,
            file_path,
            width,
            height,
            license,
            flickr_url,
            coco_url,
            date_captured,
        ) = row

        file_name = Path(file_path).name
        if file_name in target_images:
            image_ids.append(img_id)
            images.append(
                {
                    "id": img_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "license": license,
                    "flickr_url": flickr_url,
                    "coco_url": coco_url,
                    "date_captured": date_captured,
                }
            )

    return images, image_ids


def fetch_filtered_annotations(
    cursor, image_ids, target_category_ids=None, chunk_size=1000
):
    annotations = []

    for i in tqdm(
        range(0, len(image_ids), chunk_size),
        desc="Filter DB annotations",
    ):
        chunk_ids = image_ids[i : i + chunk_size]
        image_placeholders = ",".join("?" * len(chunk_ids))

        params = list(chunk_ids)
        sql = f"""
            SELECT id, image_id, category_id, bbox, area, iscrowd, segmentation
            FROM annotations
            WHERE image_id IN ({image_placeholders})
        """

        if target_category_ids:
            cat_placeholders = ",".join("?" * len(target_category_ids))
            sql += f" AND category_id IN ({cat_placeholders})"
            params.extend(target_category_ids)

        cursor.execute(sql, params)

        for ann_id, image_id, category_id, bbox, area, iscrowd, segmentation in cursor:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": json.loads(bbox),
                    "area": area,
                    "iscrowd": iscrowd,
                    "segmentation": json.loads(segmentation),
                }
            )

    return annotations


def fetch_filtered_categories(cursor, target_category_ids=None):
    categories = []
    cursor.execute("SELECT id, name, supercategory FROM categories")

    for cat_id, name, supercategory in cursor:
        if not target_category_ids or cat_id in target_category_ids:
            categories.append(
                {"id": cat_id, "name": name, "supercategory": supercategory}
            )

    return categories


def build_coco_json(images, annotations, categories):
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {"description": "Filtered COCO dataset from DB"},
        "license": [],
    }


def save_coco_json(coco_data, output_dir: Path, patch_number, split_name):
    output_dir = output_dir / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"patch_{patch_number}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)

    print(f"[OK] Saved: {output_json}")


def extract_coco_subset_from_db(
    db_path: Path,
    base_output_dir: Path,
    image_path: Path,
    patch_number: int,
    split_name="train",
    target_category_names=None,
    target_category_ids=None,
):
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    if not image_path.is_dir():
        raise FileNotFoundError(f"Image dir not found: {image_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    category_name_to_id = load_category_mapping(cursor)

    if target_category_names:
        target_category_ids = [
            category_name_to_id[name]
            for name in target_category_names
            if name in category_name_to_id
        ]

    images, image_ids = fetch_filtered_images(cursor, image_path)
    if not image_ids:
        conn.close()
        raise RuntimeError(f"No matching images found in {image_path}")

    annotations = fetch_filtered_annotations(cursor, image_ids, target_category_ids)
    categories = fetch_filtered_categories(cursor, target_category_ids)

    conn.close()

    coco_data = build_coco_json(images, annotations, categories)
    save_coco_json(coco_data, base_output_dir, patch_number, split_name)


def main():
    db_root = Path("/path/to/db/root")
    db_path = db_root / "train" / "objects365_train.db"

    patch_number = 25
    image_path = db_root / "train" / f"patch_{patch_number}" / "images"

    target_category_names = [
        "person",
        "car",
    ]

    extract_coco_subset_from_db(
        db_path=db_path,
        base_output_dir=db_root,
        image_path=image_path,
        patch_number=patch_number,
        split_name="train",
        target_category_names=target_category_names,
    )


if __name__ == "__main__":
    main()
