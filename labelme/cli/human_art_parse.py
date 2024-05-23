import argparse
import json
from labelme.utils import lmdb_util
from pathlib import Path
import os
import shutil
from tqdm import tqdm
from labelme.cli.coco2017_to_json import clean_folder
from labelme.utils import md5_util


def main(args):
    output_image_dir = Path(args.output_image_dir)
    output_ann_dir = Path(args.output_json_dir)
    # 创建目录
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    # 清理之前的输出
    clean_folder(output_image_dir)
    clean_folder(output_ann_dir)
    human_art_parent = Path(args.ann_file).parent.parent.parent
    # 读取ann_file
    with open(args.ann_file, 'r') as ann_f:
        ann_coco = json.load(ann_f)
    # 搜集需要类别的id
    select_c_ids = []
    select_c_names = []
    for ann_category_item in ann_coco["categories"]:
        if ann_category_item["name"] in args.select_categories:
            select_c_ids.append(ann_category_item["id"])
            select_c_names.append(ann_category_item["name"])
    # 遍历标记，按图片对标记分类
    image_ann_map = {}
    ignore_image_ids = []
    for ann_item in ann_coco["annotations"]:
        if ann_item["category_id"] in select_c_ids:
            # 是选中的类
            if ann_item["iscrowd"] > 0 and args.ignore_crowd:
                # 忽略crowd
                ignore_image_ids.append(ann_item["image_id"])
                continue
            if ann_item["image_id"] not in image_ann_map:
                image_ann_map[ann_item["image_id"]] = []
            image_ann_map[ann_item["image_id"]].append(ann_item)
    # 排除包含crowd的图片
    for image_id in ignore_image_ids:
        if image_id in image_ann_map:
            image_ann_map.pop(image_id)
    # 遍历图片，转换对应的标注到labelme标注
    for image_item in tqdm(ann_coco["images"]):
        image_id = image_item["id"]
        if args.only_real_human and "real_human" not in image_item["file_name"]:
            # 只处理真人图片
            continue
        if image_id in image_ann_map:
            image_path = human_art_parent.joinpath(image_item["file_name"])
            image_name = md5_util.md5_of_file(
                str(image_path))
            shutil.copyfile(image_path, output_image_dir.joinpath(
                image_name + image_path.suffix))
            shapes = []
            for ann_item in image_ann_map[image_id]:
                label_name = select_c_names[select_c_ids.index(
                    ann_item["category_id"])]
                data_item = {"label": label_name,
                             "shape_type": "rectangle",
                             "group_id": None,
                             "flags": {},
                             "description": None,
                             "mask": None}
                data_item["points"] = [[ann_item["bbox"][0], ann_item["bbox"][1]], [
                    ann_item["bbox"][0] + ann_item["bbox"][2], ann_item["bbox"][1] + ann_item["bbox"][3]]]
                shapes.append(data_item)
            json_item = {
                "version": "5.4.1",
                "shapes": shapes,
                "imagePath": "../images/" + image_name + image_path.suffix,
                "imageHeight": image_item["height"],
                "imageWidth": image_item["width"],
                "flags": None,
            }
            json_file = output_ann_dir.joinpath(image_name + '.json')
            with open(json_file, 'w') as json_f:
                json.dump(json_item, json_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='转换human art标签到labelme格式')
    parser.add_argument(
        "--ann_file", default="/home/ssbai/datas/HumanArt/annotations/training_humanart_coco.json")
    parser.add_argument("--output_image_dir", default="./output/images")
    parser.add_argument("--output_json_dir", default="./output/anns")
    parser.add_argument("--select_categories", nargs="+",
                        type=str, default=["person"])
    parser.add_argument("--ignore_crowd", default=True)
    parser.add_argument("--only_real_human", default=True)
    args = parser.parse_args()
    main(args)
