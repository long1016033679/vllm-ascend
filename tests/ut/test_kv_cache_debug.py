#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import os
from unittest import mock

from tests.ut.base import TestBase
from vllm_ascend import envs, utils


class TestKvCacheDebugHelpers(TestBase):
    def test_kv_debug_format_ids_short_list(self):
        self.assertEqual(utils.kv_debug_format_ids([1, 2, 3]), "[1, 2, 3]")
        self.assertEqual(utils.kv_debug_format_ids([]), "[]")

    def test_kv_debug_format_ids_long_list_truncated(self):
        ids = list(range(100))
        result = utils.kv_debug_format_ids(ids, max_items=8)
        self.assertIn("(共100个)", result)
        self.assertIn("[0, 1, 2, 3", result)
        self.assertIn("96, 97, 98, 99]", result)

    def test_kv_debug_format_ids_custom_max_items(self):
        ids = list(range(10))
        result = utils.kv_debug_format_ids(ids, max_items=10)
        self.assertEqual(result, str(ids))

    def test_kv_debug_format_ids_non_numeric_fallback(self):
        self.assertEqual(utils.kv_debug_format_ids(["a", "b"]), "['a', 'b']")

    def test_kv_debug_format_ids_tensor_like_input(self):
        class FakeTensorView:
            def __init__(self, values):
                self._values = values

            def __iter__(self):
                return iter(self._values)

        self.assertEqual(utils.kv_debug_format_ids(FakeTensorView([5, 6])), "[5, 6]")

    def test_is_kv_cache_debug_enabled_flag(self):
        with mock.patch.object(utils.envs_ascend, "VLLM_ASCEND_KV_DEBUG", True):
            self.assertTrue(utils.is_kv_cache_debug_enabled())
        with mock.patch.object(utils.envs_ascend, "VLLM_ASCEND_KV_DEBUG", False):
            self.assertFalse(utils.is_kv_cache_debug_enabled())

    def test_kv_debug_env_var_definition(self):
        self.assertIn("VLLM_ASCEND_KV_DEBUG", envs.env_variables)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ASCEND_KV_DEBUG", None)
            self.assertFalse(envs.env_variables["VLLM_ASCEND_KV_DEBUG"]())
        with mock.patch.dict(os.environ, {"VLLM_ASCEND_KV_DEBUG": "1"}):
            self.assertTrue(envs.env_variables["VLLM_ASCEND_KV_DEBUG"]())
        with mock.patch.dict(os.environ, {"VLLM_ASCEND_KV_DEBUG": "0"}):
            self.assertFalse(envs.env_variables["VLLM_ASCEND_KV_DEBUG"]())
