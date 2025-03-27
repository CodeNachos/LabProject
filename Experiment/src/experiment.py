import os
import json
import logging
import shutil
import numpy as np
from typing import Dict
from pathlib import Path
from enum import IntEnum
from strenum import StrEnum
from collections import defaultdict
from psychopy import visual, core, event

# CONSTANTS ====================================================================

class Modalities(IntEnum):
    MEDIUM  = 1,
    HIGH    = 2

NB_MODALITIES = 2

class _CacheKeys(StrEnum):
    SUBJECTS = 'total_subjects'
    CATIMGS = 'images_per_category',
    CATEGORIES = 'category_list'
    RANDSEED = 'random_seed'

WIN = visual.Window(monitor="testMonitor",  units="pix", fullscr=True)

_TRIAL_INSTRUCTION = "" \
"Please press the space bar as soon as you see an image appear trough the mask."

_RELATIVE_PATH = Path(__file__).resolve().parent
_STIMULI_PATH = (_RELATIVE_PATH / "../res/stimuli").resolve()
_TARGET_PATH = _STIMULI_PATH / "grayscale/"
_HIGH_MODALITY_PATH = _STIMULI_PATH / "trialready/high/"
_MEDIUM_MODALITY_PATH = _STIMULI_PATH / "trialready/medium/"

_CACHE_FILE = _RELATIVE_PATH / "../res/expdata/.expcache.json"

# default config
_DEFAULTS = {
    _CacheKeys.SUBJECTS: 0,
    _CacheKeys.CATIMGS: 4,
    _CacheKeys.CATEGORIES: (_RELATIVE_PATH / "../res/expdata/categories.txt").resolve(),
    _CacheKeys.RANDSEED: 42
}

# CONFIG =======================================================================

# logger config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


# experiment config
_subjectid = _DEFAULTS[_CacheKeys.SUBJECTS]
_nb_catimgs:int = _DEFAULTS[_CacheKeys.CATIMGS]
_categories:Path = _DEFAULTS[_CacheKeys.CATEGORIES]
_random_seed = _DEFAULTS[_CacheKeys.RANDSEED]

# UTILITY ======================================================================

# cache management

def _update_cache(key: str, value) -> None:
    data = {}

    if _CACHE_FILE.exists():
        try:
            with _CACHE_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("[WARNING]: cache file is corrupted, starting fresh.")

    data[key] = value

    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _read_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            with _CACHE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("[WARNING]: cache file is corrupted.")
    return {}

def _get_cache_value(key: str, default=None):
    return _read_cache().get(key, default)

def _load_defaults():
    if _CACHE_FILE.exists():
        os.remove(_CACHE_FILE)
    
    for key, value in _DEFAULTS.items():
        _update_cache(key.value, str(value))
    
def _load_cache():
    if not _CACHE_FILE.exists():
        logger.info("[INFO]: no existent experiment cache, loading defaults")
        _load_defaults()
        return

    global _subjectid; global _nb_catimgs
    global _categories; global _random_seed

    cache = _read_cache()

    _subjectid = int(cache[_CacheKeys.SUBJECTS])
    _nb_catimgs = int(cache[_CacheKeys.CATIMGS])
    _categories = Path(cache[_CacheKeys.CATEGORIES])
    _random_seed = int(cache[_CacheKeys.RANDSEED])


def _verify_files() -> None:
    missing_files = False
    
    logger.debug("[DEBUG]: Veryfing data integrity...")

    for path in [_TARGET_PATH, _HIGH_MODALITY_PATH, _MEDIUM_MODALITY_PATH]:
        if not path.exists():
            logger.debug(f"[DEBUG]: directory {path} doesnt exist")
            missing_files = True
            continue

        for subdir in path.iterdir():
            if subdir.is_dir():
                for category in subdir.iterdir():
                    if category.is_dir():
                        jpg_count = sum(1 for _ in category.glob("*.jpg"))
                        if jpg_count < _nb_catimgs:
                            logger.debug(f"[DEBUG]: missing images in {path}")
                            missing_files = True
    
    if not missing_files:
        logger.debug("[DEBUG]: Data integrity verified!")
        return 
    
    logger.error("[ERROR]: Missing files detected")
    logger.info("[INFO]: Recreating stimuli data...\n")

    if _STIMULI_PATH.exists(): shutil.rmtree(_STIMULI_PATH)

    import stimgrab
    stimgrab.create_stimuli(_nb_catimgs, _categories)

    logger.info("[INFO]: Stimuli data successfully generated!")


def get_images() -> Dict[str, Path]:
    images = defaultdict(list)

    subject_modalities = [m for m in Modalities]
    n = _subjectid % NB_MODALITIES
    subject_modalities = subject_modalities[-n:] + subject_modalities[:-n]

    for img_path in _TARGET_PATH.rglob("*.jpg"):
        images[img_path.parent.name].append(
            (
                img_path.name, 
                subject_modalities[len(images[img_path.parent.name])%NB_MODALITIES]
            )
        )
    
    return images

def get_fixation_cross(win=None) -> visual.ShapeStim:
    if win is None: win=WIN

    return visual.ShapeStim(
        win, vertices='cross', 
        size=(win.size[0]*.01, win.size[1]*.01*(win.size[0]/win.size[1])), 
        fillColor='black', lineColor=None)

def get_mondrian_pattern(
        nb_rectangles, 
        pattern_scale=(1,1), 
        rect_scale=0.1,
        color=None,
        max_opacity=None,
        win=None) -> visual.ElementArrayStim:

    if win is None: win=WIN
    
    rect_size = np.array([win.size[0]*rect_scale, win.size[1]*rect_scale*3]) * pattern_scale
    rectangles = []
    for i in range(nb_rectangles):
        if color is None:
            rand_shade = np.random.uniform(-1, 0)
            rect_color = [rand_shade, rand_shade, rand_shade]
        else:
            rect_color = color
        rect = visual.Rect(
            win, 
            size=rect_size*np.random.uniform(.6, 1),
            fillColor=rect_color, 
            lineColor=None,
            pos=(
                np.random.randint(
                    -(win.size[0]/2) * pattern_scale[0], 
                    (win.size[0]/2) * pattern_scale[1]
                ), np.random.randint(
                    -(win.size[1]/2) * pattern_scale[0], 
                    (win.size[1]/2) * pattern_scale[1]
                )
            ),
            opacity= 1 if max_opacity is None else np.random.uniform(0, max_opacity)
        )
        rectangles.append(rect)

    return visual.ElementArrayStim(
        win, units="pix", nElements=nb_rectangles, 
        elementTex=None, elementMask=None, 
        xys=[rect.pos for rect in rectangles], 
        sizes=[rect.size for rect in rectangles], 
        colors=[rect.fillColor for rect in rectangles],
        opacities=[rect.opacity for rect in rectangles]
    )

# FLOW =========================================================================

def show_instructions(win=None):
    if win is None: win=WIN

    instruction_text = visual.TextStim(
        win,
        height=30,
        color="white",
        text=_TRIAL_INSTRUCTION
    )

    continue_text = visual.TextStim(
        win, 
        bold=True,
        height=20,
        color="white", 
        pos=(.0, -win.size[1]*0.3),
        text="Press [space bar] key to contiue..."
    )
    
    frame = visual.Rect(
        win, 
        fillColor=None, lineColor='white' ,
        size=(win.size[0]*.85, win.size[1]*.85)
    )
    
    instruction_text.draw()
    continue_text.draw()
    frame.draw()

    win.flip()

    event.waitKeys(keyList=['space'])


# MAIN EXECUTION ===============================================================

if __name__ == "__main__":
    _load_cache()
    _categories:Path = (_RELATIVE_PATH / "../test.txt").resolve()
    _verify_files()
    images = get_images()
    show_instructions()
    background = get_mondrian_pattern(200)
    while True:
        background.draw()
        get_fixation_cross().draw()
        WIN.flip()
        event.waitKeys(keyList=['space'])
        _update_cache(_CacheKeys.SUBJECTS, _subjectid+1)
        break
    WIN.close()      
    core.quit()