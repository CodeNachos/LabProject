import os
import json
import shutil
import logging
import argparse
import stimgrab
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
    REDFILTER = 'red_filter'
    CYANFILTER = 'cyan_filter',
    USERANDSEED = 'use_random_seed'
    RANDSEED = 'random_seed',

WIN = visual.Window(monitor="testMonitor",  units="pix", fullscr=True, allowStencil=True)

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
    _CacheKeys.SUBJECTS     : 0,
    _CacheKeys.CATIMGS      : 4,
    _CacheKeys.CATEGORIES   : (_RELATIVE_PATH / "../res/expdata/categories.txt").resolve(),
    _CacheKeys.REDFILTER    : [-.75, -.1, -.1],
    _CacheKeys.CYANFILTER   : [-.1, -.7, -.7],
    _CacheKeys.USERANDSEED  : True,
    _CacheKeys.RANDSEED     : 42
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

# main program parser
parser = argparse.ArgumentParser(
    prog="Experiment",
    description="Runs the experimental pipeline, including data collection, processing, and result generation."
)
parser.add_argument(
    '--version', 
    action='version', 
    version='%(prog)s 0.0.1'
)
parser.add_argument(
    '-r', '--reset', 
    action='store_true', 
    help="Reset settings and cache to default values"
)
parser.add_argument(
    '-s', '--set-subjects',
    type=int,
    help="Set number of total past subjects"
)
parser.add_argument(
    '-i', '--images-per-category',
    type=int,
    help="Set number of images show per category"
)
parser.add_argument(
    '-c', '--categories',
    type=str,
    help="Path to a category list text file"
)
parser.add_argument(
    '--cyan_filter',
    type=list,
    help="Set cyan filter color"
)
parser.add_argument(
    '--red-filter',
    type=list,
    help="Set red filter color"
)
parser.add_argument(
    '--seed',
    type=int,
    help="Set random seed"
)
parser.add_argument(
    '--use-seed',
    action='set-false',
    help="If given it won't use a constant random seed"
)
parser.add_argument(
    '-v', '--verbose',
    action="store_false",
    help="Increase output verbosity"
)

# experiment config
_subjectid = _DEFAULTS[_CacheKeys.SUBJECTS]
_nb_catimgs:int = _DEFAULTS[_CacheKeys.CATIMGS]
_categories:Path = _DEFAULTS[_CacheKeys.CATEGORIES]
_red_filter = _DEFAULTS[_CacheKeys.REDFILTER]
_cyan_filter = _DEFAULTS[_CacheKeys.CYANFILTER]
_random_seed = _DEFAULTS[_CacheKeys.RANDSEED]
_use_random_seed = _DEFAULTS[_CacheKeys.USERANDSEED]

# UTILITY ======================================================================

# manage command line argumentss

def _handle_args():
    args = parser.parse_args()
    
    for key, value in vars(args).items():
        if value is not None:
            if key == "reset":
                if value: _load_defaults()
            elif key == "set_subjects":
                _update_cache(_CacheKeys.SUBJECTS, value)
            elif key == "images-per-category":
                _update_cache(_CacheKeys.CATIMGS, value)
            elif key == "categories":
                _update_cache(_CacheKeys.CATEGORIES, value)
            elif key == "red-filter":
                _update_cache(_CacheKeys.REDFILTER, value)
            elif key == "cyan-filter":
                _update_cache(_CacheKeys.CYANFILTER, value)
            elif key == "seed":
                _update_cache(_CacheKeys.RANDSEED, value)
            elif key == 'use-seed':
                _update_cache(_CacheKeys.USERANDSEED, value)
            elif key == "verbose":
                if value: 
                    logger.setLevel(logging.INFO)
                    logging.getLogger("stimgrab").setLevel(logging.INFO)
                else: 
                    logger.setLevel(logging.ERROR)
                    logging.getLogger("stimgrab").setLevel(logging.ERROR)
                    print  (logging.getLogger("stimgrab"))
            
# cache management

def _json_serialize(value):
    if isinstance(value, Path):
        return str(value)
    return value

def _update_cache(key: str, value) -> None:
    data = {}

    if _CACHE_FILE.exists():
        try:
            with _CACHE_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("[WARNING]: cache file is corrupted, starting fresh.")

    data[key] = _json_serialize(value)

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
        _update_cache(key.value, value)
    
def _load_cache():
    if not _CACHE_FILE.exists():
        logger.info("[INFO]: no existent experiment cache, loading defaults")
        _load_defaults()
        return

    global _subjectid; global _nb_catimgs; global _categories;
    global _random_seed; global _use_random_seed;
    global _red_filter; global _cyan_filter;

    cache = _read_cache()

    _subjectid = int(cache[_CacheKeys.SUBJECTS])
    _nb_catimgs = int(cache[_CacheKeys.CATIMGS])
    _categories = Path(cache[_CacheKeys.CATEGORIES])
    _red_filter = cache[_CacheKeys.REDFILTER]
    _cyan_filter = cache[_CacheKeys.CYANFILTER]
    _random_seed = int(cache[_CacheKeys.RANDSEED])
    _use_random_seed = bool(cache[_CacheKeys.USERANDSEED])

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
                img_path, 
                subject_modalities[len(images[img_path.parent.name])%NB_MODALITIES]
            )
        )
    
    return images

# VISUAL =======================================================================

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

def run_cycle():
    pass

def run_trial():
    background = get_mondrian_pattern(200)

    stimuli = get_images()



def main():
    _handle_args()
    _load_cache()
    _verify_files()

    np.random.seed(_random_seed)


if __name__ == "__main__":
    _handle_args()
    _load_cache()
    _verify_files()

    images = get_images()
    show_instructions()
    
    background = get_mondrian_pattern(200)
    stimulus_frame = visual.Circle(WIN, size=WIN.size[0]*.5, color="gray")
    color_filter = np.array(_red_filter)


    clock = core.Clock()
    opacity = 1
    start_time = clock.getTime()

    flash_rate = 3  # Hz
    flash_interval = 1.0 / flash_rate  # seconds
    duration = 6  # seconds for the whole CFS trial

    while clock.getTime() - start_time < duration:
        # Get the current time
        current_time = clock.getTime()

        # Update opacity (smoothly decreases over time)
        opacity = (current_time / duration)

        background.draw()
        stimulus_frame.draw()

        aperture = visual.Aperture(WIN, size=WIN.size[0]*.4, shape='circle')
        aperture.enabled = True
        image = visual.ImageStim(
            win=WIN,
            image=images['bridge-68'][_subjectid%_nb_catimgs][0],
            size=WIN.size[0]*.4
        )
        
        image.setColor(color_filter)
        image.opacity = opacity
        image.draw()
        mask = get_mondrian_pattern(200, (.7,.7), color=_cyan_filter, max_opacity=1-opacity)
        mask.draw()
        aperture.enabled = False

        get_fixation_cross().draw()
        WIN.flip()

        # Check for vkey press during the loop
        keys = event.getKeys(timeStamped=clock)  # Capture key presses with timestamps
        if keys:
            response_time = keys[0][1]  # Capture the time at which the key was pressed
            print(f"Response Time: {response_time:.4f} seconds")
            break
        
        elapsed_time = clock.getTime() - current_time  # Time since last frame update
        core.wait(flash_interval - elapsed_time)

    
    _update_cache(_CacheKeys.SUBJECTS, _subjectid+1)
    
    WIN.close()
    core.quit()