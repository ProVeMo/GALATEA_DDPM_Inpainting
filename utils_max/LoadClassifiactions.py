
import yaml
import os

import copy

def load_classifiactions(path: str, conditions : dict = None):
    if os.path.exists(path) and not os.path.isfile(path):
        path = os.path.join(path, 'classifications.yaml')


    if os.path.isfile(path):
        with open(path, 'r') as file:
            # load annotations
            # body part:
            #    vid0001: [ [0,42], [67,123] ]
            #    vid0002: [ [0, 42], [67, 123], [560,700] ]
            #    ....
            # next_body_part:
            #   ....
            annots = yaml.safe_load(file)
            regions_blank = annots['regions_blank'] # possible body parts in vid frames
            annots_list = annots['annotations']
            filtered_annotations = copy.deepcopy(annots_list)
            for region in annots_list: # get every body part
                # load only relevant images
                if conditions is None or region not in conditions.keys():
                    classes = ['normal']
                    for c in annots_list[region]:
                        if c != 'normal':
                            filtered_annotations[region].pop(c, None)
                # for conditional load only configured classes e.g ulcus, erosion...
                else:
                    classes = conditions[region]
                    #if 'normal' not in classes:
                        #classes.append('normal')
                    for c in annots_list[region]:
                        if c not in classes:
                            filtered_annotations[region].pop(c, None)

                # parse conditions e.g ulcus, erosion... for every body part
                for condition in classes:
                    # get every video for each condition
                    for video in annots_list[region][condition]:
                        # get all region slices (only idx of good imgs)
                        for i, roi in enumerate(annots_list[region][condition][video]):
                            if len(roi) > 1:  # check if whole region is defined and not just a single img
                                filtered_annotations[region][condition][video][i] = list(range(roi[0], roi[1]))

            return regions_blank, filtered_annotations
    else:
        raise FileNotFoundError(f"Couldnt find classifications.yaml file at {path}")




if __name__ == '__main__':
    data_path = '../../dataset/tempDataset_v2/'
    print(os.listdir(data_path))
    trainA = os.path.join(data_path, 'trainA', )
    trainB = os.path.join(data_path, 'trainB')
    regions_blank, annots = load_classifiactions(trainB)
    print(annots['mund']['normal'].items() )
    #print(type(annots['annotations']['ausserhalb']['Dummy01-01-0013'][0]) )
