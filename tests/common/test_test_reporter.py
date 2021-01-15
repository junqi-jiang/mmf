# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.test_reporter import TestReporter


class TestTestReporter(TestReporter):
    def __init__(self, dataset, dataloader):
        self.current_dataset_idx = -1
        self.datasets = [dataset]
        self.test_dataloader = dataloader
        self.report = []

    def get_dataloader(self):
        return self.test_dataloader
