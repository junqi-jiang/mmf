# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import copy
import warnings
from collections import OrderedDict

import torch
from mmf.common.sample import detach_tensor


class Report(OrderedDict):
    def __init__(self, batch=None, model_output=None, *args):
        super().__init__(self)
        if batch is None:
            return
        if model_output is None:
            model_output = {}
        if self._check_and_load_tuple(batch):
            return

        all_args = [batch, model_output] + [*args]
        for idx, arg in enumerate(all_args):
            if not isinstance(arg, collections.abc.Mapping):
                raise TypeError(
                    "Argument {:d}, {} must be of instance of "
                    "collections.abc.Mapping".format(idx, arg)
                )

        self.batch_size = batch.get_batch_size()
        self.warning_string = (
            "Updating forward report with key {}"
            "{}, but it already exists in {}. "
            "Please consider using a different key, "
            "as this can cause issues during loss and "
            "metric calculations."
        )

        for idx, arg in enumerate(all_args):
            for key, item in arg.items():
                if key in self and idx >= 2:
                    log = self.warning_string.format(
                        key, "", "in previous arguments to report"
                    )
                    warnings.warn(log)
                self[key] = item

    def get_batch_size(self):
        return self.batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def _check_and_load_tuple(self, batch):
        if isinstance(batch, collections.abc.Mapping):
            return False

        if isinstance(batch[0], (tuple, list)) and isinstance(batch[0][0], str):
            for kv_pair in batch:
                self[kv_pair[0]] = kv_pair[1]
            return True
        else:
            return False

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self):
        return list(self.keys())

    def apply_fn(self, fn, fields=None):
        for key in self.keys():
            if fields is not None and isinstance(fields, (list, tuple)):
                if key not in fields:
                    continue
            self[key] = fn(self[key])
            if isinstance(self[key], collections.MutableSequence):
                for idx, item in enumerate(self[key]):
                    self[key][idx] = fn(item)
            elif isinstance(self[key], dict):
                for subkey in self[key].keys():
                    self[key][subkey] = fn(self[key][subkey])
        return self

    def detach(self):
        return self.apply_fn(detach_tensor)

    def to(self, device, non_blocking=True, fields=None):
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError(
                    "device must be either 'str' or "
                    "'torch.device' type, {} found".format(type(device))
                )
            device = torch.device(device)

        def fn(x):
            if hasattr(x, "to"):
                x = x.to(device, non_blocking=non_blocking)
            return x

        return self.apply_fn(fn, fields)

    def accumulate_tensor_fields_and_loss(self, report, field_list):
        for key in field_list:
            if key == "__prediction_report__":
                continue
            if key not in self.keys():
                warnings.warn(
                    f"{key} not found in report. Metrics calculation "
                    + "might not work as expected."
                )
                continue
            if isinstance(self[key], torch.Tensor):
                self[key] = torch.cat((self[key], report[key]), dim=0)

        self._accumulate_loss(report)

    def _accumulate_loss(self, report):
        for key, value in report.losses.items():
            if key not in self.losses.keys():
                warnings.warn(
                    f"{key} not found in report. Loss calculation "
                    + "might not work as expected."
                )
                continue
            if isinstance(self.losses[key], torch.Tensor):
                self.losses[key] += value

    def copy(self):
        """Get a copy of the current Report

        Returns:
            SampleList: Copy of current Report.

        """
        report = Report()

        fields = self.fields()

        for field in fields:
            report[field] = copy.deepcopy(self[field])

        return report
