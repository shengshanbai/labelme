import argparse
from pathlib import Path
import cv2
import json
from tqdm import tqdm


def main(args):
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for txt_file in tqdm(in_dir.iterdir()):
        if txt_file.suffix == '.txt':
            item_name = txt_file.stem
            image_path = image_dir.joinpath(item_name + ".jpg")
            image_np = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            h, w = image_np.shape[:2]
            # 读取txt文件
            shapes = []
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line_parts = line.strip().split()
                    x_center = float(line_parts[1]) * w
                    y_center = float(line_parts[2]) * h
                    obj_w = float(line_parts[3]) * w
                    obj_h = float(line_parts[4]) * h
                    data_item = {"label": args.label,
                                 "shape_type": "rectangle",
                                 "group_id": None,
                                 "flags": {},
                                 "description": None,
                                 "mask": None}
                    data_item["points"] = [[x_center - obj_w / 2, y_center -
                                           obj_h / 2], [x_center + obj_w / 2, y_center + obj_h / 2]]
                    shapes.append(data_item)
            # 绘制shape到image上
            # for shape in shapes:
            #     cv2.rectangle(image_np, (int(shape["points"][0]), int(shape["points"][1])),
            #                   (int(shape["points"][2]), int(shape["points"][3])), (0, 255, 255), 2)
            # cv2.imwrite("./temp.jpg", image_np)
            json_item = {
                "version": "5.4.1",
                "shapes": shapes,
                "imagePath": "../images/" + image_path.name,
                "imageHeight": h,
                "imageWidth": w,
                "flags": None,
            }
            # 创建json文件
            json_file = out_dir.joinpath(item_name + '.json')
            with open(json_file, 'w') as json_f:
                json.dump(json_item, json_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("转换yolo格式数据到labelme的json格式")
    parser.add_argument("--label", default="human", type=str, help="类别名")
    parser.add_argument(
        "--image_dir", default="/home/ssbai/datas/detection/train/images", type=str, help="输入的图片路径")
    parser.add_argument(
        "--input_dir", default="/home/ssbai/datas/human_dataset/labels/train", type=str, help="输入的yolo格式数据")
    parser.add_argument(
        "--output_dir", default="/home/ssbai/datas/detection/train/anns", type=str, help="输出的json格式数据")
    args = parser.parse_args()
    main(args)
