import os
import sys
import torch
import functools
import intel_extension_for_pytorch  # noqa:F401
from .fake_module import common, nccl
from collections import namedtuple
from ruamel.yaml import YAML
from .wrap_api import WrapAPI
from typing import Optional, Any
from torch.types import _dtype

yaml = YAML(typ="safe", pure=True)
not_callable_list = [
    "is_bf16_supported",
    "has_half",
    "_initialization_lock",
    "_initialized",
    "_lazy_seed_tracker",
    "_queued_calls",
    "_tls",
    "threading",
    "traceback",
]

pre_device_class = torch.device


def get_yaml_list(file_path: str):
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path),
        "r",
        encoding="utf-8",
    ) as f:
        yaml_list = yaml.load(f.read())
    return (
        yaml_list["tensor_to_api"],
        yaml_list["create_tensor_supported_api"],
        yaml_list["data_loader_api"],
        yaml_list["ddp_api"],
        yaml_list["pass_api"],
        yaml_list["failure_api"],
        yaml_list["ccl_api"],
        yaml_list["create_tensor_unsupported_api"],
    )


def get_api_info(dist_backend):
    api_map = namedtuple("api_entry", "api_mod api_name api_wrap")

    (
        to_list,
        create_tensor_list,
        data_loader_list,
        ddp_list,
        pass_list,
        failure_list,
        ccl_list,
        create_tensor_unsupported_list,
    ) = get_yaml_list("yaml/register_support_api.yaml")

    torch_api_map_list = []
    api_list_supported = (
        to_list
        + create_tensor_list
        + data_loader_list
        + ddp_list
        + pass_list
        + ccl_list
        + failure_list
    )
    api_list_unsupported = create_tensor_unsupported_list

    api_list = api_list_supported + api_list_unsupported
    for item in api_list:
        item_list = item.split(".")
        mod = item_list[:-1]
        length = len(mod)
        new_mod = item_list[0]
        for i in range(1, length):
            new_mod = new_mod + "." + item_list[i]

        eval_new_mod = new_mod
        if new_mod != "":
            eval_new_mod = eval(new_mod)
        api_name = item_list[-1]
        if item in pass_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_pass)
            )
        elif item in failure_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_failure)
            )
        elif item in ccl_list:
            if dist_backend == "ccl":
                torch_api_map_list.append(
                    api_map(eval_new_mod, api_name, WrapAPI.wrap_api_ccl)
                )
        elif item in to_list:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_to)
            )
        elif item in api_list_unsupported:
            if api_name not in not_callable_list:
                torch_api_map_list.append(
                    api_map(eval_new_mod, api_name, WrapAPI.wrap_api_skip)
                )
        else:
            torch_api_map_list.append(
                api_map(eval_new_mod, api_name, WrapAPI.wrap_api_common)
            )

    return torch_api_map_list


def get_attr(mod, name):
    api = None
    try:
        api = getattr(mod, name)
    except AttributeError:
        pass
    return api


def set_attr(mod, name, new_name):
    try:
        setattr(mod, name, new_name)
    except AttributeError:
        pass


class device_meta_class(type):
    def __instancecheck__(cls, instance):
        if instance is None:
            return False
        return isinstance(instance, pre_device_class)


class fake_device(metaclass=device_meta_class):
    def __new__(cls, device_item, i=-1):
        # device can be device_type, device_index, device_type:device_item
        # deal with torch.device
        if isinstance(device_item, pre_device_class):
            if device_item.type == "xpu" or device_item.type == "cpu":
                return pre_device_class(device_item.type, device_item.index)
            elif device_item.type == "cuda":
                return pre_device_class("xpu", device_item.index)
            elif device_item.type == "meta":
                return pre_device_class("meta", device_item.index)
            else:
                raise RuntimeError(
                    "[Compatible mode] Met unexpected device type when creating new device object",
                    device_item.type,
                )

        # special case for only index, torch will use cuda device
        if isinstance(device_item, int):
            return pre_device_class("xpu", device_item)

        # met string here, may be cuda:0 or cuda
        # need to check torch.xpu.current_device to get index
        device_item = device_item.replace("cuda", "xpu")

        # prevent bad fork process go through here
        if i == -1 and device_item.find(":") == -1 and not torch.xpu._is_in_bad_fork():
            current_device = torch.xpu.current_device()
            if current_device != 0:
                i = current_device

        return (
            pre_device_class(device_item)
            if i == -1
            else pre_device_class(device_item, i)
        )

    def __reduce__(self):
        return (self.__class__, (self.type, self.index))

def apply_patch(module_name, func_name, target_func):
    original_module = sys.modules[module_name]
    original_get_gpu_type = getattr(original_module, func_name)
    original_module.get_gpu_type = target_func

    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and hasattr(mod, func_name):
            if getattr(mod, func_name) == original_get_gpu_type:
                setattr(mod, func_name, target_func)


# here is corner case for tensor.cuda(), it will call .to method without device args()
def fake_cuda(tensor_input, index=-1, non_blocking=None):
    if isinstance(index, str):
        index = index.replace("cuda", "xpu")
    # if non_blocking is bool, it means caller use this args
    if isinstance(non_blocking, bool):
        return (
            tensor_input.xpu(index, non_blocking=non_blocking)
            if index != -1
            else tensor_input.xpu(non_blocking=non_blocking)
        )
    return tensor_input.xpu(index) if index != -1 else tensor_input.xpu()


def is_cuda(input_args):
    return input_args.is_xpu

class WrapHelper:
    def __init__(
        self, target_device="xpu", dist_backend="ccl", compile_backend="inductor"
    ):
        self.torch_api_map = set()
        self.target_device = target_device
        self.dist_backend = dist_backend
        self.compile_backend = compile_backend

    def convert_api(self):
        if self.target_device == "xpu":
            torch_api_map_list = get_api_info(self.dist_backend)
            for item in torch_api_map_list:
                self.torch_api_map.add(item)
            for item in self.torch_api_map:
                api = get_attr(item.api_mod, item.api_name)

                # handle the interface for torch.Tensor.cuda and torch.nn.Module.cuda
                if item.api_name == "cuda":
                    api = get_attr(item.api_mod, "to")

                if api is not None:
                    set_attr(item.api_mod, item.api_name, item.api_wrap(api))

            # disable torch.jit.script for cannot support
            set_attr(torch.jit, "script", WrapAPI.wrap_jit_script(torch.jit.script))

            # fake for torch.cuda.amp.common for it cannot be found in torch.xpu
            set_attr(torch.cuda, "nccl", nccl)
            set_attr(torch.cuda.amp, "common", common)
            set_attr(
                torch.cuda,
                "GradScaler",
                functools.partial(torch.amp.GradScaler, device="xpu"),
            )

        torch_autocast = torch.autocast

        class fake_autocast:
            def __init__(
                self,
                device_type="xpu",
                dtype: Optional[_dtype] = None,
                enabled: bool = True,
                cache_enabled: Optional[bool] = None,
            ):
                if device_type.find("cuda") != -1:
                    self.i = torch_autocast(
                        device_type="xpu",
                        dtype=torch.bfloat16,
                        enabled=enabled,
                        cache_enabled=cache_enabled,
                    )

                self.i = torch_autocast(
                    device_type="xpu",
                    dtype=torch.bfloat16,
                    enabled=enabled,
                    cache_enabled=cache_enabled,
                )
                self.i.fast_dtype = torch.bfloat16

            def __enter__(self):
                self.i.__enter__()

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
                self.i.__exit__(exc_type, exc_val, exc_tb)

            def __call__(self, func):
                return self.i.__call__(func)

        torch.autocast = fake_autocast
        torch.cuda.amp.autocast = fake_autocast

        def fake_get_gpu_type():
            return "xpu"

        # Tma is for nvidia hopper, we didn't have the counter part
        def has_triton_tma():
            return False

        import triton
        from triton.backends import backends as triton_backends

        def xpu_create_driver():
            actives = [
                x.driver for x in triton_backends.values() if x.driver.is_active()
            ]
            for idx, active in enumerate(actives):
                if "XPUDriver" == active.__name__:
                    return actives[idx]()
            raise RuntimeError(
                "Could not found triton for xpu, pls install triton for XPU"
            )

        def avoid_device_init(self) -> bool:
            if torch.xpu._is_compiled():
                return not torch.xpu.is_available()

            return not (
                torch.cuda.is_available()
                or (hasattr(torch, "hpu") and torch.hpu.is_available())
            )

        torch._subclasses.fake_tensor.FakeTensorMode.avoid_device_init = (
            avoid_device_init
        )

        def set_driver_to_gpu():
            driver = triton.runtime.driver
            for name, backend in triton.backends.backends.items():
                # TODO: here should abstract 'intel' to a device name if we will support other device
                if backend.driver.is_active() and name != "cpu" and "intel" == name:
                    if isinstance(driver.active, backend.driver):
                        # Don't re-initialize backend if it is already active
                        return
                    driver.set_active(backend.driver())
                    return
            raise RuntimeError("Could not find an active GPU backend")

        # quiet ugly but has no choice to init the defaultConfig driver only with xpu
        triton.runtime.driver.default.__init__(xpu_create_driver)

        apply_patch("torch._inductor.utils", "get_gpu_type", fake_get_gpu_type)
        apply_patch("torch.utils._triton", "has_triton_tma", has_triton_tma)
        apply_patch(
            "torch._inductor.runtime.triton_helpers",
            "set_driver_to_gpu",
            set_driver_to_gpu,
        )

        set_attr(torch.Tensor, "cuda", fake_cuda)
        set_attr(torch.nn.Module, "cuda", fake_cuda)
        set_attr(torch.Tensor, "is_cuda", is_cuda)
        set_attr(torch.nn.Module, "is_cuda", is_cuda)

    def convert_var(self):
        if self.target_device == "xpu":
            torch.has_cuda = True
            torch.cuda.has_half = True
            torch.version.cuda = "11.7"
            torch._C._XpuDeviceProperties.major = 8
            torch._C._XpuDeviceProperties.minor = 5

            # set device property
            device_property = torch.xpu.get_device_properties(torch.device("xpu"))
            torch._C._XpuDeviceProperties.multi_processor_count = (
                device_property.gpu_subslice_count
            )
            torch.cuda.amp.GradScaler = torch.amp.GradScaler

            torch.device = fake_device
            # TODO: major, minor. Major means the arch, minor means the incremental imporvement

    def convert_module(self):
        def replace_backend(target_backend, replace_backend, name):
            if name.startswith(target_backend):

                migrate_name = replace_backend + name[len(target_backend) :]
                if migrate_name in sys.modules.keys():
                    sys.modules[name] = sys.modules[migrate_name]

        if self.target_device == "xpu":
            for name, mod in sys.modules.items():
                replace_backend("torch.cuda", "torch.xpu", name)
                replace_backend(
                    "torch.backends.cuda",
                    "intel_extension_for_pytorch.backends.xpu",
                    name,
                )

            torch.cuda = sys.modules["torch.cuda"]
            torch.backends.backends = sys.modules["torch.backends.cuda"]


def compatible_mode(
    target_device="xpu", dist_backend="ccl", compile_backend="inductor"
):
    helper = WrapHelper(target_device, dist_backend, compile_backend)

    # convert torch function outside of module [torch.cuda, torch.backends.cuda]
    helper.convert_module()
    # convert torch apis using device or set "cuda" device as default device
    helper.convert_var()
    # convert torch attr related with cuda device
    helper.convert_api()
