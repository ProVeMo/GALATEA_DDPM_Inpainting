import json
import os
import sys


def classification2Json(path: str = None):
    if os.path.exists(os.path.join(path, "classification.py")):
        sys.path.append(path)
        from classification import annot, annot2, regions_blank
        json_data = {
            'regions_blank': regions_blank,
            'annot': annot,
            'annot2': annot2
        }
        with open("classification.json", "w") as outfile:
            json.dump(json_data, outfile)


def Json2classification(path: str = None):
    if path:
        classificationObjects = []
        with open(path) as json_file:
            data = json.load(json_file)

        regions_blank = data['regions_blank']
        annot = data['annot']
        annot2 = data['annot2']

        return regions_blank, annot, annot2

    else:
        assert False, "Json2classification: Path muss angegeben werden"


if __name__ == '__main__':
    classification2Json("/home/mnielsen/Documents/Workspace/tempGAN/dataset/tempDataset/trainB")
    # regio_ratio, annot, annot2 = Json2classification("C:\\Users\\maxi_\\Documents\\MasterarbeitGAN\\workspace\\KISandbx\\utils\\classification.json")
    print("classification2Json.main() wurde ausgeführt")
