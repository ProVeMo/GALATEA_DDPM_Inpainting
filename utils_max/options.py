import yaml
import os
import datetime


class options():
    def __init__(self, path="config-defaults.yaml"):
        self.dir = os.path.dirname(__file__)
        self.osOpts = load_options(os.path.join(self.dir, "options.yaml"))
        self.path = path
        with open(path) as f:
            self.data = yaml.load(f, Loader=yaml.FullLoader)


    def __call__(self, key):
        return self.data[key]['value']

    def __iadd__(self, other):
        self.osOpts["run_id"] += other
        path = os.path.join(self.dir, "options.yaml")
        with open(path, 'w') as f:
            yaml.dump(self.osOpts, f)
        return self

    def setVal(self, key, val):
        self.data[key]['value'] = val
        with open(self.path, 'w') as f:
            yaml.dump(self.data, f)

    # Operation system specific settings:
    def os(self, val):
        return self.osOpts[val]


def load_options(path="config-defaults.yaml"):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data


def save_options(options):
    with open('options.yaml', 'w') as f:
        yaml.dump(options, f)


def show(options):
    for key, val in options.items():
        if isinstance(val, datetime.date):
            # print(key + ' : ' + val.)
            print("Hat nicht geklappt")
        else:

            print(key + ' : ' + val)
