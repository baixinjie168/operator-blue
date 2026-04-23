# -*- coding: UTF-8 -*-
"""
功能：定义数据生成脚本共用的数据模型
"""
from enum import Enum


class DispatcherTargetType(Enum):
    METHOD = "method"
    CLASS = "class"
