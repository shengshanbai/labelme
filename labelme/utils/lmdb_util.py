import lmdb
import os
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import json

PROTOCOL_LEVEL = 4


def read_lmdb(root_dir):
    db_env = lmdb.open(root_dir, readonly=True, lock=False)
    return db_env


def write_lmdb(root_dir, map_size=None):
    if map_size is not None:
        db_env = lmdb.open(root_dir, map_size=map_size)
    else:
        db_env = lmdb.open(
            root_dir,
        )
    return db_env


def read_folder(db_env, path=""):
    # 忽略结束的"/"
    if path.endswith("/"):
        path = path[: len(path) - 1]
    with db_env.begin() as txn:
        subs = txn.get("{}/__keys__".format(path).encode())
        if subs is None:
            return None
        subs = pickle.loads(subs)
    return subs


def read_txt(db_env, path):
    with db_env.begin() as txn:
        file_content = txn.get(path.encode())
        file_content = pickle.loads(file_content)
        file_content = file_content.decode("utf-8")
    return file_content


def read_json(db_env, path):
    with db_env.begin() as txn:
        file_content = txn.get(path.encode())
        file_content = pickle.loads(file_content)
        file_content = file_content.decode("utf-8")
        json_c = json.loads(file_content)
    return json_c


def read_image(db_env, path, grayscale=False):
    # return bgr image if not grayscale
    with db_env.begin() as txn:
        file_content = txn.get(path.encode())
        file_content = pickle.loads(file_content)
        image = cv2.imdecode(
            np.frombuffer(file_content, np.uint8),
            cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR,
        )
        if not grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_binary(db_env, path):
    with db_env.begin() as txn:
        file_content = txn.get(path.encode())
        file_content = pickle.loads(file_content)
    return file_content


def write_file(db_env, prefix, f_name, file_folder):
    f_key = f_name
    with open(os.path.join(file_folder, f_name), "rb") as f:
        f_content = f.read()
    try:
        with db_env.begin(write=True) as txn:
            txn.put(
                "{}/{}".format(prefix, f_key).encode(),
                pickle.dumps(f_content, protocol=4),
            )
    except lmdb.MapFullError:
        db_env.set_mapsize(db_env.info()["map_size"] + 1024 * 1024 * 1024)
        with db_env.begin(write=True) as txn:
            txn.put(
                "{}/{}".format(prefix, f_key).encode(),
                pickle.dumps(f_content, protocol=4),
            )


def write_keys(db_env, prefix, keys):
    try:
        with db_env.begin(write=True) as txn:
            txn.put(
                "{}/__keys__".format(prefix).encode(),
                pickle.dumps(keys, protocol=PROTOCOL_LEVEL),
            )
    except lmdb.MapFullError:
        db_env.set_mapsize(db_env.info()["map_size"] + 512 * 1024 * 1024)
        with db_env.begin(write=True) as txn:
            txn.put(
                "{}/__keys__".format(prefix).encode(),
                pickle.dumps(keys, protocol=PROTOCOL_LEVEL),
            )


def write_folder(db_env, folder, prefix="", folder_filter=None):
    sub_items = os.listdir(folder)
    if folder_filter is not None:
        sub_items = list(filter(folder_filter, sub_items))
    write_keys(db_env, prefix, sub_items)
    for sub_item in tqdm(sub_items):
        if os.path.isdir(os.path.join(folder, sub_item)):
            sub_prefix = "{}/{}".format(prefix, sub_item)
            write_folder(db_env, os.path.join(folder, sub_item), sub_prefix)
        else:
            write_file(db_env, prefix, sub_item, folder)


def write_sample(db_env, sample, index):
    try:
        with db_env.begin(write=True) as txn:
            txn.put(f"{index}".encode(), pickle.dumps(
                sample, protocol=PROTOCOL_LEVEL))
    except lmdb.MapFullError:
        db_env.set_mapsize(db_env.info()["map_size"] + 1024 * 1024 * 1024)
        with db_env.begin(write=True) as txn:
            txn.put(f"{index}".encode(), pickle.dumps(
                sample, protocol=PROTOCOL_LEVEL))


def write_file_content(db_env, f_key, f_content):
    try:
        with db_env.begin(write=True) as txn:
            txn.put(f_key.encode(), pickle.dumps(f_content, protocol=4))
    except lmdb.MapFullError:
        db_env.set_mapsize(db_env.info()["map_size"] + 1024 * 1024 * 1024)
        with db_env.begin(write=True) as txn:
            txn.put(f_key.encode(), pickle.dumps(f_content, protocol=4))
