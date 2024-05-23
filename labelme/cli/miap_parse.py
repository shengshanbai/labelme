import argparse
import json
from labelme.utils import lmdb_util
from pathlib import Path
import os
import shutil
from tqdm import tqdm
from labelme.cli.coco2017_to_json import clean_folder
from labelme.utils import md5_util
import cv2


def main(args):
    output_image_dir = Path(args.output_image_dir)
    output_ann_dir = Path(args.output_json_dir)
    # 创建目录
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    # 清理之前的输出
    clean_folder(output_image_dir)
    clean_folder(output_ann_dir)
    image_dir = Path(args.image_dir)
    # 按行读取ann file
    image_anns_map = {}
    ignore_image_ids = []
    with open(args.ann_file, "r") as ann_f:
        # 打印并跳过第一行
        line = ann_f.readline()
        print(line)
        while True:
            line = ann_f.readline()
            if not line:
                break
            line_parts = line.strip().split(",")
            image_id = line_parts[0]
            x_min = float(line_parts[3])
            x_max = float(line_parts[4])
            y_min = float(line_parts[5])
            y_max = float(line_parts[6])
            is_group_of = int(line_parts[9])
            is_depiction_of = int(line_parts[10])
            # 忽略一群的标注
            if is_group_of > 0 and args.ignore_crowd:
                ignore_image_ids.append(image_id)
                continue
            if image_id not in image_anns_map:
                image_anns_map[image_id] = []
            image_anns_map[image_id].append({
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "is_group_of": is_group_of,
                "is_depiction_of": is_depiction_of,
            })
    # 排除包含crowd的图片
    for image_id in ignore_image_ids:
        if image_id in image_anns_map:
            image_anns_map.pop(image_id)
    # 遍历图片
    for image_id in tqdm(image_anns_map.keys()):
        image_name = image_id + ".jpg"
        image_path = image_dir.joinpath(image_name)
        out_name = md5_util.md5_of_file(image_path)
        shutil.copyfile(
            image_path, output_image_dir.joinpath(out_name + ".jpg"))
        image_np = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_h, image_w = image_np.shape[:2]
        shapes = []
        for ann_item in image_anns_map[image_id]:
            data_item = {"label": "person",
                         "shape_type": "rectangle",
                         "group_id": None,
                         "flags": {},
                         "description": None,
                         "mask": None}
            data_item["points"] = [[ann_item["x_min"] * image_w, ann_item["y_min"] * image_h], [
                ann_item["x_max"] * image_w, ann_item["y_max"] * image_h]]
            shapes.append(data_item)
        json_item = {
            "version": "5.4.1",
            "shapes": shapes,
            "imagePath": "../images/" + out_name + ".jpg",
            "imageHeight": image_h,
            "imageWidth": image_w,
            "flags": None,
        }
        json_file = output_ann_dir.joinpath(out_name + '.json')
        with open(json_file, 'w') as json_f:
            json.dump(json_item, json_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='转换MIAP标签到labelme格式')
    parser.add_argument(
        "--ann_file", default="/media/ssbai/d/datas/miap/open_images_extended_miap_boxes_val.csv")
    parser.add_argument(
        "--image_dir", default="/media/ssbai/d/datas/miap/images/val")
    parser.add_argument("--output_image_dir", default="./output/images")
    parser.add_argument("--output_json_dir", default="./output/anns")
    parser.add_argument("--select_categories", nargs="+",
                        type=str, default=["person"])
    parser.add_argument("--ignore_crowd", default=True)
    args = parser.parse_args()
    main(args)
