# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import tasks, criterions, models  # noqa
import os, importlib
# for file in sorted(os.listdir(os.path.dirname(__file__))):
#     if file.endswith(".py") and not file.startswith("_"):
#         task_name = file[: file.find(".py")]
#         print(task_name)
#         importlib.import_module("examples.asr_st_mt.tasks." + task_name)