from __future__ import annotations

from enum import Enum


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    EXTRACT = "extract"
    WAIT = "wait"
    SELECT = "select"
    PRESS_KEY = "press_key"
