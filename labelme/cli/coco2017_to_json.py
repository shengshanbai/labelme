import argparse
import json
from labelme.utils import lmdb_util
from pathlib import Path
import os
import shutil
from tqdm import tqdm


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def main(args):
    output_image_dir = Path(args.output_image_dir)
    output_ann_dir = Path(args.output_json_dir)
    # 创建目录
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    # 清理之前的输出
    clean_folder(output_image_dir)
    clean_folder(output_ann_dir)
    input_db_env = lmdb_util.read_lmdb(args.input_db)
    ann_json = lmdb_util.read_json(input_db_env, args.ann_file)
    # 搜集需要类别的id
    select_c_ids = []
    select_c_names = []
    for ann_category_item in ann_json["categories"]:
        if ann_category_item["name"] in args.select_categories:
            select_c_ids.append(ann_category_item["id"])
            select_c_names.append(ann_category_item["name"])
    # 遍历标记，按图片对标记分类
    image_ann_map = {}
    ignore_image_ids = []
    for ann_item in ann_json["annotations"]:
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
    for image_item in tqdm(ann_json["images"]):
        image_id = image_item["id"]
        if image_id in image_ann_map:
            image_name = image_item['file_name']
            image_binary = lmdb_util.read_binary(
                input_db_env, f"{args.image_dir}/{image_name}")
            with open(output_image_dir.joinpath(image_name), "wb") as image_f:
                image_f.write(image_binary)
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
                "imagePath": "../images/" + image_name,
                "imageHeight": image_item["width"],
                "imageWidth": image_item["height"],
                "flags": None,
            }
            json_file = output_ann_dir.joinpath(
                os.path.splitext(image_name)[0] + '.json')
            with open(json_file, 'w') as json_f:
                json.dump(json_item, json_f)
    input_db_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("转换coco2017的标注到labelme的标注")
    parser.add_argument("--input_db", type=str,
                        default="/home/ssbai/datas/coco2017_db")
    parser.add_argument("--ann_file", type=str,
                        default="/annotations/instances_train2017.json")
    parser.add_argument("--image_dir", type=str, default="/train2017")
    parser.add_argument("--select_categories", nargs="+",
                        type=str, default=["person"])
    parser.add_argument("--ignore_crowd", default=True)
    parser.add_argument("--output_image_dir", default="./output/images")
    parser.add_argument("--output_json_dir", default="./output/anns")
    args = parser.parse_args()
    main(args)
