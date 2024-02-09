# -*- coding: utf-8 -*-
from datatransform import DataTransform
from datetime import datetime, date
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='Transform original data to TFRecord')

parser.add_argument('--label', type=str, default='click')
parser.add_argument("--store_stat", action="store_true")
parser.add_argument("--threshold", type=int, default=10)
parser.add_argument("--dataset", type=Path, default="./avazu_new/train.csv")
parser.add_argument("--record", type=Path, default="./avazu_new/threshold_")
parser.add_argument("--ratio", nargs='+', type=float, default=[0.8, 0.1, 0.1])

args = parser.parse_args()
if args.record == Path("./avazu_new/threshold_"):
    args.record = Path("./avazu_new/threshold_"+str(args.threshold)+"/")
os.makedirs(args.record, exist_ok=True)


class AvazuTransform(DataTransform):
    def __init__(self, dataset_path, path, min_threshold, label_index, ratio, store_stat=False, seed=2022):
        super(AvazuTransform, self).__init__(dataset_path, path, store_stat=store_stat, seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.split = ratio
        self.path = path

    def process(self):
        self._read(name=None, header=0, sep=",", label_index=self.label)
        if self.store_stat:
            self.generate_and_filter(threshold=self.threshold, label_index=self.label)
        tr, val, te = self.random_split(ratio=self.split)
        self.transform_tfrecord(tr, self.path, "train", label_index=self.label)
        self.transform_tfrecord(val, self.path, "test", label_index=self.label)
        self.transform_tfrecord(te, self.path, "validation", label_index=self.label)

    def _process_x(self):
        hour = self.data["hour"].apply(lambda x: str(x))

        def _convert_weekday(time):
            dt = date(int("20" + time[0:2]), int(time[2:4]), int(time[4:6]))
            return int(dt.strftime("%w"))

        self.data["weekday"] = hour.apply(_convert_weekday)

        def _convert_weekend(time):
            dt = date(int("20" + time[0:2]), int(time[2:4]), int(time[4:6]))
            return 1 if dt.strftime("%w") in ['6', '0'] else 0

        self.data["is_weekend"] = hour.apply(_convert_weekend)

        self.data["hour"] = hour.apply(lambda x: int(x[6:8]))

    def _process_y(self):
        self.data = self.data.drop("id", axis=1)


if __name__ == "__main__":
    tranformer = AvazuTransform(args.dataset, args.record, args.threshold, args.label, args.ratio, args.store_stat, 2022)
    tranformer.process()
