import os
import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm


def extract_coco_subset_from_db(
    db_path: Path,
    base_output_dir: Path,
    image_path: Path,
    patch_number: int,
    split_name: str = "train",
    target_category_names: Optional[Iterable[str]] = None,
    target_category_ids: Optional[Iterable[int]] = None,
):
    """
    Extract a COCO-style subset JSON from a SQLite DB,
    filtered by existing image files and optional target categories.

    Args:
        db_path: Path to SQLite DB file
        base_output_dir: Base output directory
        image_path: Directory containing target images
        patch_number: Patch index for output file naming
        split_name: train/val/test
        target_category_names: Optional list of category names
        target_category_ids: Optional list of category ids
    """

    db_path = Path(db_path)
    base_output_dir = Path(base_output_dir)
    image_path = Path(image_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Cannot find the database file: {db_path}")

    if not image_path.is_dir():
        raise NotADirectoryError(f"Cannot find the image folder: {image_path}")

    target_images = set(os.listdir(image_path))

    filtered_image_info: List[dict] = []
    filtered_image_ids: List[int] = []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    category_name_to_id = {}
    cursor.execute("SELECT id, name FROM categories")
    for cat_id, name in cursor:
        category_name_to_id[name] = cat_id

    if target_category_names:
        resolved_ids = []
        for name in target_category_names:
            if name not in category_name_to_id:
                print(f"[WARN] Category name not found in DB: {name}")
                continue
            resolved_ids.append(category_name_to_id[name])

        target_category_ids = resolved_ids

    if target_category_ids:
        target_category_ids = list(set(target_category_ids))

    cursor.execute(
        """
        SELECT id, file_name, width, height, license,
               flickr_url, coco_url, date_captured
        FROM images
        """
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
            filtered_image_ids.append(img_id)
            filtered_image_info.append(
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

    if not filtered_image_ids:
        conn.close()
        raise RuntimeError(f"Cannot find any images in {image_path} from the database")

    filtered_annotations: List[dict] = []
    chunk_size = 1000

    for i in tqdm(
        range(0, len(filtered_image_ids), chunk_size),
        desc="Filter DB annotations",
    ):
        chunk_ids = filtered_image_ids[i : i + chunk_size]
        image_placeholders = ",".join("?" * len(chunk_ids))

        params = list(chunk_ids)
        sql = f"""
            SELECT id, image_id, category_id,
                   bbox, area, iscrowd, segmentation
            FROM annotations
            WHERE image_id IN ({image_placeholders})
        """

        if target_category_ids:
            cat_placeholders = ",".join("?" * len(target_category_ids))
            sql += f" AND category_id IN ({cat_placeholders})"
            params.extend(target_category_ids)

        cursor.execute(sql, params)

        for row in cursor:
            ann_id, image_id, category_id, bbox, area, iscrowd, segmentation = row

            try:
                bbox_parsed = json.loads(bbox)
            except Exception:
                bbox_parsed = []

            try:
                segmentation_parsed = json.loads(segmentation)
            except Exception:
                segmentation_parsed = []

            filtered_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox_parsed,
                    "area": area,
                    "iscrowd": iscrowd,
                    "segmentation": segmentation_parsed,
                }
            )

    processed_categories: List[dict] = []
    cursor.execute("SELECT id, name, supercategory FROM categories")

    for cat_id, name, supercategory in cursor:
        if not target_category_ids or cat_id in target_category_ids:
            processed_categories.append(
                {
                    "id": cat_id,
                    "name": name,
                    "supercategory": supercategory,
                }
            )

    conn.close()

    new_coco_data = {
        "images": filtered_image_info,
        "annotations": filtered_annotations,
        "categories": processed_categories,
        "info": {"description": "Filtered COCO dataset from DB"},
        "licenses": [],
    }

    output_dir = base_output_dir / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"patch_{patch_number}.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(new_coco_data, f, indent=4, ensure_ascii=False)

    print(f"[OK] Saved: {output_json}")
