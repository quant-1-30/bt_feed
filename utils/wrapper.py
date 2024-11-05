import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from .loader import get_module_by_module_path 


#################### Wrapper #####################
class Wrapper:
    """Wrapper class for anything that needs to set up during qlib.init"""

    def __init__(self):
        self._provider = None

    def register(self, provider):
        self._provider = provider

    def __repr__(self):
        return "{name}(provider={provider})".format(name=self.__class__.__name__, provider=self._provider)

    def __getattr__(self, key):
        if self.__dict__.get("_provider", None) is None:
            raise AttributeError("Please run qlib.init() first using qlib")
        return getattr(self._provider, key)


def register_wrapper(wrapper, cls_or_obj, module_path=None):
    """register_wrapper

    :param wrapper: A wrapper.
    :param cls_or_obj:  A class or class name or object instance.
    """
    if isinstance(cls_or_obj, str):
        module = get_module_by_module_path(module_path)
        cls_or_obj = getattr(module, cls_or_obj)
    obj = cls_or_obj() if isinstance(cls_or_obj, type) else cls_or_obj
    wrapper.register(obj)
    