import hashlib


def md5(buffer):
    hash_md5 = hashlib.md5()
    hash_md5.update(buffer)
    return hash_md5.hexdigest()


def md5_of_file(file):
    with open(file, 'rb') as fh:
        m = hashlib.md5()
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)
    return m.hexdigest()
