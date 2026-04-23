"""
actions.py — Action Engine
Maps gesture labels to system-level or simulated actions.

NOTE FOR DEPLOYMENT:
  pyautogui actions (scroll, screenshot, key press) work on a local machine.
  On cloud deployments (Hugging Face Spaces) they are silently skipped and
  a log message is printed instead — the rest of the demo still works fully.
"""

import time
import logging

logger = logging.getLogger(__name__)

# Try to import pyautogui; if it fails (headless server) use a stub
try:
    import pyautogui
    pyautogui.FAILSAFE = False   # disable corner-failsafe for demos
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("pyautogui not available — actions will be logged only.")

# ── Action Definitions ────────────────────────────────────────────────────────

def _scroll_up():
    if PYAUTOGUI_AVAILABLE:
        pyautogui.scroll(5)
    logger.info("ACTION: scroll_up")

def _scroll_down():
    if PYAUTOGUI_AVAILABLE:
        pyautogui.scroll(-5)
    logger.info("ACTION: scroll_down")

def _screenshot():
    if PYAUTOGUI_AVAILABLE:
        ts = int(time.time())
        pyautogui.screenshot(f"screenshot_{ts}.png")
    logger.info("ACTION: screenshot")

def _volume_up():
    if PYAUTOGUI_AVAILABLE:
        pyautogui.press("volumeup")
    logger.info("ACTION: volume_up")

def _volume_down():
    if PYAUTOGUI_AVAILABLE:
        pyautogui.press("volumedown")
    logger.info("ACTION: volume_down")

def _play_pause():
    if PYAUTOGUI_AVAILABLE:
        pyautogui.press("playpause")
    logger.info("ACTION: play_pause")

def _next_tab():
    if PYAUTOGUI_AVAILABLE:
        pyautogui.hotkey("ctrl", "tab")
    logger.info("ACTION: next_tab")

def _prev_tab():
    if PYAUTOGUI_AVAILABLE:
        pyautogui.hotkey("ctrl", "shift", "tab")
    logger.info("ACTION: prev_tab")

def _zoom_in():
    if PYAUTOGUI_AVAILABLE:
        pyautogui.hotkey("ctrl", "+")
    logger.info("ACTION: zoom_in")

def _zoom_out():
    if PYAUTOGUI_AVAILABLE:
        pyautogui.hotkey("ctrl", "-")
    logger.info("ACTION: zoom_out")

def _noop():
    logger.info("ACTION: none (no-op gesture)")

# Registry: action name → function
ACTION_REGISTRY = {
    "scroll_up":    _scroll_up,
    "scroll_down":  _scroll_down,
    "screenshot":   _screenshot,
    "volume_up":    _volume_up,
    "volume_down":  _volume_down,
    "play_pause":   _play_pause,
    "next_tab":     _next_tab,
    "prev_tab":     _prev_tab,
    "zoom_in":      _zoom_in,
    "zoom_out":     _zoom_out,
    "none":         _noop,
}

def execute_action(action_name: str):
    """Execute an action by name. Unknown names are logged and ignored."""
    fn = ACTION_REGISTRY.get(action_name)
    if fn:
        fn()
    else:
        logger.warning(f"Unknown action: {action_name}")

def available_actions() -> list:
    """Return list of all registered action names."""
    return list(ACTION_REGISTRY.keys())
