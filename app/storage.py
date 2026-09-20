import json


def save_metadata(chunks, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )


def load_metadata(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)