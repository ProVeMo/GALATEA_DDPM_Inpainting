# Sampling for training Dataset "tempDataset"
import os
import random
import re
import copy
import natsort
from typing import Tuple, Dict
from classification2Json import Json2classification
from LoadClassifiactions import load_classifiactions

# Memory intensive
class FrameSampler():
    def __init__(self, folder_path: str, frame_interval: int = 3, region_ratio: Dict = None, conditional_classes : dict = None):
        self.folder_path = folder_path
        self.frame_interval = frame_interval
        self.conditional_classes = conditional_classes
        self.regions_blank, self.annot2 = load_classifiactions( os.path.join(folder_path, "classifications.yaml"), conditions=self.conditional_classes )

        if ( self.regions_blank is not None and self.annot2 is not None):
            # get rid json and .py representations for frame annotations
                #self.regions_blank, self.annot, self.annot2 = Json2classification(
                #os.path.join(folder_path, "classification.json"))
            self.annot = None # not sure if this will be needed in future or annot2 is enough
            forbidden_files_list = ['classification.py', 'classification.json', '__pycache__', 'classification.yaml']
            self.vid_list = os.listdir(self.folder_path)
            # remove file which doesnt belong to actual videos
            for file in forbidden_files_list:
                if file in self.vid_list:
                    self.vid_list.remove(file)

            self.classification_exists = True
        else:
            print(
                f"{self.__class__.__name__}.__init__(): classifications.yaml konnte in {folder_path} nicht gefunden werden.")
            self.annot = None
            self.annot2 = None
            self.regions_blank = None
            self.classification_exists = False

        if self.classification_exists:
            if region_ratio:
                self.region_ratio = region_ratio
            else:
                self.region_ratio, self.region_ratio_rel = self.__calculate_region_ratio_classification(folder_path)
        else:
            self.__calculate_targetframe_list()

    def __calculate_region_ratio_classification(self, folder_path: str) -> Tuple[Dict, Dict]:
        # region_ratio_abs
        region_ratio = copy.deepcopy(self.regions_blank)
        for vid in self.vid_list:
            path = os.path.join(folder_path, vid)
            for idx, frame_name in enumerate(os.listdir(path)):
                frame = int(re.split('(\d+)', frame_name)[1])
                change = False
                for region, frame_list in self.annot[vid].items():
                    if frame in frame_list:
                        region_ratio[region] += 1
                        change = True
                if not change:
                    region_ratio['none'] += 1

        # region_ratio_rel
        sum = 0
        for key, value in region_ratio.items():
            sum += value
        region_ratio_rel = copy.deepcopy(self.regions_blank)
        for region, number in region_ratio.items():
            region_ratio_rel[region] = round(number / sum, 4)

        return region_ratio, region_ratio_rel

    def __calculate_targetframe_list(self):
        self.number_frames_in_folder = []
        for dir in os.listdir(self.folder_path):
            self.number_frames_in_folder.append(len(os.listdir(os.path.join(self.folder_path, dir))))
        self.target_frame_list = []
        for vid in os.listdir(self.folder_path):
            vid_path = os.path.join(self.folder_path, vid)
            frame_list = natsort.natsorted(os.listdir(vid_path))
            for _ in range(self.frame_interval):
                frame_list.pop(-1)
            for frame in frame_list:
                self.target_frame_list.append(vid + ";" + frame)

    def __get_random_from_weigthed_region(self, region_ratio: dict = None) -> str:
        if not region_ratio:
            return random.choices(list(self.region_ratio.keys()), weights=list(self.region_ratio.values()))[0]
        else:
            return random.choices(list(region_ratio.keys()), weights=list(region_ratio.values()))[0]

    def __get_random_targetframe_from_region(self, region: str = None) -> dict:
        if not region:
            region = self.__get_random_from_weigthed_region()
        condition_dict = self.annot2[region]
        # in case of conditional GAN: choose a random condition (index) like ulcus, erosion...
        condition_keys = list(condition_dict.keys())
        if self.conditional_classes == None or region not in self.conditional_classes:
            condition = condition_keys[0]
        else:
            rand_idx = random.randint(0, len(self.conditional_classes[region]) - 1)
            condition = self.conditional_classes[region][rand_idx]

        number_of_possible_frame_sequences_abs = 0
        number_of_possible_frame_sequences_list = []


        for vid, frames_list in condition_dict[condition].items():
            for subsequence in frames_list:
                for i in range(len(subsequence) - (self.frame_interval)):
                    number_of_possible_frame_sequences_list.append(vid + ';' + str(subsequence[i]))
        random_seq = random.choice(number_of_possible_frame_sequences_list)
        target_vid = random_seq.split(';')[0]
        target_frame = int(random_seq.split(';')[1])

        return {'video' : target_vid, 'target_frame': target_frame, 'region': region, 'condition': condition}

    def __get_random_targetframe(self):
        random_seq = random.choice(self.target_frame_list)
        target_vid = random_seq.split(';')[0]
        target_frame = int(re.split('(\d+)', random_seq.split(';')[1])[1])
        return target_vid, target_frame

    def get_targetframe(self, region: str = None) -> Tuple[str, int]:
        if self.classification_exists:
            if region:
                return self.__get_random_targetframe_from_region(region=region)
            else:
                return self.__get_random_targetframe_from_region()
        else:
            return self.__get_random_targetframe()


def main():
    """
    regions = ['none', 'ausserhalb', 'mund', 'kehlkopf', 'oseophagus', 'z_linie', 'magen', 'pylorus', 'duodenum',
               'inversion']
    region_ratio_trainB_blank = {'none': 0, 'ausserhalb': 0, 'mund': 0, 'kehlkopf': 0, 'oseophagus': 0,
                                 'z_linie': 0, 'magen': 0, 'pylorus': 0, 'duodenum': 0, 'inversion': 0}
    region_ratio_trainA_blank = {'none': 1}
    # calculated with FrameSampler.calculate_region_ratio()
    region_ratio_abs = {'none': 2968, 'ausserhalb': 2886, 'mund': 1332, 'kehlkopf': 1554, 'oseophagus': 10068,
                        'z_linie': 6827, 'magen': 21480, 'pylorus': 5414, 'duodenum': 9530, 'inversion': 2500}
    region_ratio_norm = {'none': 0.046, 'ausserhalb': 0.0447, 'mund': 0.0206, 'kehlkopf': 0.0241, 'oseophagus': 0.156,
                         'z_linie': 0.1057, 'magen': 0.3327, 'pylorus': 0.0839, 'duodenum': 0.1476, 'inversion': 0.0387}
    """
    import time

    region_ratio = {'none': 1, 'ausserhalb': 3000, 'mund': 1500, 'kehlkopf': 2000, 'oseophagus': 15000,
                    'z_linie': 6000, 'magen': 15000, 'pylorus': 6000, 'duodenum': 950, 'inversion': 3500}

    # example:
    path_B = "F:\\Masterarbeit\\Datasets\\tempDataset\\trainB"
    path_A = "F:\\Masterarbeit\\Datasets\\tempDataset\\trainA"

    start = time.time()
    fs_A = FrameSampler(folder_path=path_A)
    print('fs_A_init time: ', time.time() - start)

    start = time.time()
    fs_B = FrameSampler(folder_path=path_B, region_ratio=region_ratio)
    print('fs_B_init time: ', time.time() - start)

    start = time.time()
    fs_B.get_targetframe()
    print('fs_B.get_targetframe(): ', time.time() - start)

    start = time.time()
    fs_A.get_targetframe()
    print('fs_A.get_targetframe(): ', time.time() - start)

    print("ads")


if __name__ == '__main__':
    main()
