import torch
from torch.testing._internal.common_utils import TestCase

import torch_comp
torch_comp.compatible_mode()

cuda_device = torch.device("cuda")


class TestTorchMethod(TestCase):
    def test_has_cuda(self):
        x = torch.has_cuda
        self.assertEqual(x, True)

    def test_version_cuda(self):
        x = torch.version.cuda
        self.assertEqual(x, "11.7")
