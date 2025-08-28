"""
Device package for photonic component design and fabrication.

This package provides classes and functions for creating various photonic devices
including cavities, couplers, and complete structures from YAML configurations.

Main Classes:
    Structure: Main class for building complete photonic structures from YAML configs

Cavity Functions:
    ring: Create ring resonator components
    race_track: Create racetrack resonator components  
    spiral: Create spiral resonator components

Coupler Functions:
    point_coupler: Create point couplers
    symmetric_pulley_coupler: Create symmetric pulley couplers
    out_point_coupler: Create output point couplers

Usage:
    from device import Structure
    
    # Load structure from config file
    structure = Structure.from_config_id("config", "1")
    component = structure.build()
    structure.plot()
    structure.save_gds("output.gds")
"""
from __future__ import annotations
import importlib
import pkgutil
import threading
from typing import Any, Dict, Protocol
import inspect
import gdsfactory as gf

from .basics import anchor_arc, straight

pdk = gf.get_active_pdk()
pdk.cells["anchor_arc"] = anchor_arc
pdk.cells["straight"] = straight

class _Provider(Protocol):
    def __call__(self, **cfg): ...

def _ensure_callable(obj: Any) -> _Provider:
    # 函数
    if inspect.isfunction(obj):
        return obj
    # 类：实例可调用 或 有 build()
    if inspect.isclass(obj):
        inst = obj()
        if callable(inst):
            return inst
        if hasattr(inst, "build"):
            return getattr(inst, "build")
        raise TypeError(f"{obj} 不是可调用，也没有 build()")
    # 对象：可调用 或 有 build()
    if callable(obj):
        return obj
    if hasattr(obj, "build"):
        return getattr(obj, "build")
    raise TypeError(f"无法将 {obj} 适配为可调用 provider")

class _Registry:
    def __init__(self, name: str):
        self._name = name
        self._map: Dict[str, _Provider] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, name: str, obj: Any, *, aliases: list[str] | None = None, overwrite: bool = False):
        if (name in self._map) and not overwrite:
            raise KeyError(f"{self._name}: 重复注册 '{name}'")
        self._map[name] = _ensure_callable(obj)
        for al in (aliases or []):
            self._aliases[al] = name

    def resolve(self, name: str) -> _Provider:
        key = self._aliases.get(name, name)
        if key not in self._map:
            candidates = ", ".join(sorted(self._map.keys()))
            raise KeyError(f"{self._name}: 未找到 '{name}'。可用: [{candidates}]")
        return self._map[key]

    def names(self):
        return sorted(self._map.keys())

# 两个全局注册表
device_registry = _Registry("device_registry")
coupler_registry = _Registry("coupler_registry")

# 函数/类 的注册装饰器
def register_device(name: str, *, aliases: list[str] | None = None, overwrite: bool = False):
    def deco(obj):
        device_registry.register(name, obj, aliases=aliases, overwrite=overwrite)
        return obj
    return deco

def register_coupler(name: str, *, aliases: list[str] | None = None, overwrite: bool = False):
    def deco(obj):
        coupler_registry.register(name, obj, aliases=aliases, overwrite=overwrite)
        return obj
    return deco

# ---- 类方法注册：类内方法写 @as_device("short_name")，类上写 @register_device_class ----
def as_device(name: str):
    """标记类中的某个方法为 device 提供者；配合 @register_device_class 使用。"""
    def deco(fn):
        setattr(fn, "_device_name_", name)
        return fn
    return deco

def register_device_class(cls):
    """扫描类中标记过 @as_device 的方法，注册为短名。"""
    for attr, val in cls.__dict__.items():
        dev_name = getattr(val, "_device_name_", None)
        if not dev_name:
            continue
        # 支持实例方法 / classmethod / staticmethod
        if isinstance(val, classmethod):
            fn = val.__func__
            def _provider_factory(func):
                def _provider(**cfg):
                    return func(cls, **cfg)
                return _provider
            provider = _provider_factory(fn)
        elif isinstance(val, staticmethod):
            fn = val.__func__
            def _provider(**cfg):
                return fn(**cfg)
            provider = _provider
        else:
            # 普通实例方法
            def _provider_factory(method_name):
                def _provider(**cfg):
                    inst = cls()
                    return getattr(inst, method_name)(**cfg)
                return _provider
            provider = _provider_factory(attr)

        device_registry.register(dev_name, provider)
    return cls

_DISCOVERED = False
_LOCK = threading.Lock()

def ensure_discovered():
    """扫描并导入 device 包下所有子模块，触发装饰器注册。"""
    global _DISCOVERED
    if _DISCOVERED:
        return
    with _LOCK:
        if _DISCOVERED:
            return
        for m in pkgutil.walk_packages(__path__, __name__ + "."):
            # 导入 device.* 的所有模块（cavity、archimedean、coupler、phc、mechanics…）
            importlib.import_module(m.name)
        _DISCOVERED = True

__all__ = [
    # 注册/解析相关
    "register_device", "register_coupler",
    "device_registry", "coupler_registry",
    "as_device", "register_device_class",
    "ensure_discovered"
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Photonic device design and fabrication tools"
