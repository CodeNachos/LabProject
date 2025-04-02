import os
import csv
import json
import shutil
import logging
import argparse
import stimgrab
import numpy as np
from time import time
from typing import Dict
from pathlib import Path
from enum import IntEnum
from itertools import chain
from strenum import StrEnum
from collections import defaultdict
from psychopy import visual, core, event

# CONSTANTS ====================================================================

class Modalities(IntEnum):
    CONTROL = 0
    LOW = 1
    MEDIUMLOW = 2
    MEDIUM = 3
    HIGH = 4

    def __str__(self):
        names_str = {
            Modalities.CONTROL   : "control",
            Modalities.LOW       : "low",
            Modalities.MEDIUMLOW : "medium-low",
            Modalities.MEDIUM    : "medium",
            Modalities.HIGH      : "high"
        }
        return names_str[self]

NB_MODALITIES = 5

class Languages(StrEnum):
    EN = "en"
    FR = "fr"

class _CacheKeys(StrEnum):
    SUBJECTS = 'total_subjects'
    CATIMGS = 'images_per_category'
    CATEGORIES = 'category_list'
    REDFILTER = 'red_filter'
    CYANFILTER = 'cyan_filter'
    USERANDSEED = 'use_random_seed'
    RANDSEED = 'random_seed'
    LANGUAGE = Languages.EN

_TRIAL_INSTRUCTION = {
    Languages.EN:
        "Welcome!\n"
        "In each trial, you will be presented with a cue followed by a short delay, "
        "and then a flickering pattern will appear on the screen.\n"
        "At some point, an image will slowly begin to emerge through the mask.\n\n"
        "Your task:\n"
        "As soon as you start to see an image, press the space bar as quickly and accurately as possible.\n\n"
        "- Keep your eyes on the center of the screen throughout the trial.\n"
        "- Press the space bar only when you are sure you have started to see the image breaking through the mask.\n"
        "- There are no correct or incorrect images; we are only measuring when you become aware of it.\n\n"
        "Take your time, stay focused, and do your best.\n\n"
        "There are 3 different cue modalities in this experiment: control, textual, and visual.\n"
        "- In the control modality, no predictive cue will be shown — only a screen indicating the modality.\n"
        "- In the textual modality, a word or a sentence will be displayed that may help you anticipate the image.\n"
        "- In the visual modality, a picture related to the upcoming image will be displayed.\n\n"
        "Try to pay attention to the cues, as they may help you detect the image faster.\n\n"
        "Thank you for participating!\n\n\n"
        "Press the space bar to continue to the trials...",

    Languages.FR:
        "Bienvenue !\n"
        "À chaque essai, un indice vous sera présenté, suivi d'un court délai, "
        "puis un motif clignotant apparaîtra à l'écran.\n"
        "Une image commencera alors à émerger progressivement à travers ce masque.\n\n"
        "Votre tâche :\n"
        "Dès que vous commencez à voir une image, appuyez sur la barre d'espace le plus rapidement et précisément possible.\n\n"
        "- Gardez vos yeux fixés au centre de l'écran pendant toute la durée de l'essai.\n"
        "- Appuyez sur la barre d'espace uniquement si vous êtes sûr(e) de commencer à voir une image apparaître à travers le masque.\n"
        "- Il n'y a pas de bonne ou de mauvaise image ; nous mesurons uniquement le moment où vous en prenez conscience.\n\n"
        "Prenez votre temps, restez concentré(e) et faites de votre mieux.\n\n"
        "Il y a 3 modalités d'indices différentes dans cette expérience : contrôle, textuelle et visuelle.\n"
        "- Dans la modalité contrôle, aucun indice prédictif ne sera affiché — seulement un écran indiquant la modalité.\n"
        "- Dans la modalité textuelle, un mot ou une phrase sera affiché et pourra vous aider à anticiper l'image.\n"
        "- Dans la modalité visuelle, une image liée à l'image cible sera présentée comme indice.\n\n"
        "Essayez de prêter attention aux indices, car ils peuvent vous aider à détecter l'image plus rapidement.\n\n"
        "Merci pour votre participation !\n\n\n"
        "Appuyez sur la barre d'espace pour continuer vers les essais..."
}

_RELATIVE_PATH = Path(__file__).resolve().parent
_STIMULI_PATH = (_RELATIVE_PATH / "../res/stimuli").resolve()
_TARGET_PATH = _STIMULI_PATH / "grayscale/"
_HIGH_MODALITY_PATH = _STIMULI_PATH / "trialready/high/"
_MEDIUM_MODALITY_PATH = _STIMULI_PATH / "trialready/medium/"
_MEDIUMLOW_MODALITY_PATH = _STIMULI_PATH / "trialready/mediumlow/"
_LOW_MODALITY_PATH = _STIMULI_PATH / "trialready/low/"

_CACHE_FILE = (_RELATIVE_PATH / "../res/expdata/.cache/expcache.json").resolve()

_TRIAL_DATA = (_RELATIVE_PATH / "../res/expdata/trials/").resolve()

# default config
_DEFAULTS = {
    _CacheKeys.SUBJECTS     : 0,
    _CacheKeys.CATIMGS      : 5,
    _CacheKeys.CATEGORIES   : (_RELATIVE_PATH / "../res/expdata/categories.txt").resolve(),
    _CacheKeys.REDFILTER    : [-.75, -.1, -.1],
    _CacheKeys.CYANFILTER   : [-.1, -.7, -.7],
    _CacheKeys.USERANDSEED  : False,
    _CacheKeys.RANDSEED     : 42,
    _CacheKeys.LANGUAGE     : Languages.EN

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
    action='store_true',
    help="If specified it will use a specified seed for the random number generation"
)
parser.add_argument(
    '-l', '--language',
    choices=['en', 'fr'],
    default='en',
    help='Language for output: "en" for English, "fr" for French (default: en)'
)
parser.add_argument(
    '--create-stimuli',
    action='store_true',
    help="If specified it will create stimuli if necessary"
)
parser.add_argument(
    '--windowed',
    action='store_true',
    help="If specified will run the experiment in windowed mode."
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
_use_seed = _DEFAULTS[_CacheKeys.USERANDSEED]
_language = _DEFAULTS[_CacheKeys.LANGUAGE]

_create_stimuli = False
_windowed = False

# CLASSES ======================================================================

class SubjectLog:
    
    __headers =  ["target", "modality", "RT"]

    def __init__(self, subjectid=None):
        logfilename = f"subject_{_subjectid if subjectid is None else subjectid}.csv"
        logfilepath = _TRIAL_DATA / logfilename

        _TRIAL_DATA.mkdir(parents=True, exist_ok=True)

        self.__logfile = open(logfilepath, mode='w', newline='', encoding='utf-8')
        self.__writer = csv.writer(self.__logfile)
        self.__writer.writerow(self.__headers)

    def log(self, target_path, modality, response_time):
        """append a new row to the CSV log file."""
        if response_time is None:
            response_time = -1
        self.__writer.writerow([target_path, modality, response_time])

    def close(self):
        """close the file when done."""
        self.__logfile.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


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
            elif key == "images_per_category":
                _update_cache(_CacheKeys.CATIMGS, value)
            elif key == "categories":
                _update_cache(_CacheKeys.CATEGORIES, value)
            elif key == "red_filter":
                _update_cache(_CacheKeys.REDFILTER, value)
            elif key == "cyan_filter":
                _update_cache(_CacheKeys.CYANFILTER, value)
            elif key == "seed":
                _update_cache(_CacheKeys.RANDSEED, value)
            elif key == 'use_seed':
                _update_cache(_CacheKeys.USERANDSEED, value)
            elif key == "language":
                _update_cache(_CacheKeys.LANGUAGE, value)
            elif key == 'create_stimuli':
                global _create_stimuli
                _create_stimuli = value
            elif key == 'windowed':
                global _windowed
                _windowed = value
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
    global _random_seed; global _use_seed; global _language;
    global _red_filter; global _cyan_filter;

    cache = _read_cache()

    _subjectid = int(cache[_CacheKeys.SUBJECTS])
    _nb_catimgs = int(cache[_CacheKeys.CATIMGS])
    _categories = Path(cache[_CacheKeys.CATEGORIES])
    _red_filter = cache[_CacheKeys.REDFILTER]
    _cyan_filter = cache[_CacheKeys.CYANFILTER]
    _random_seed = int(cache[_CacheKeys.RANDSEED])
    _use_seed = bool(cache[_CacheKeys.USERANDSEED])
    _language = cache[_CacheKeys.LANGUAGE]

# data/files utility
 
def _verify_files() -> None:
    missing_files = False
    
    logger.debug("[DEBUG]: Veryfing data integrity...")

    for path in [
        _TARGET_PATH, 
        _HIGH_MODALITY_PATH, 
        _MEDIUM_MODALITY_PATH, 
        _MEDIUMLOW_MODALITY_PATH,
        _LOW_MODALITY_PATH
    ]:
        if not path.exists():
            logger.debug(f"[DEBUG]: directory {path} doesnt exist")
            missing_files = True
            continue

        for subdir in path.iterdir():
            if subdir.is_dir():
                for category in subdir.iterdir():
                    if category.is_dir():
                        if path in [
                            _TARGET_PATH, 
                            _HIGH_MODALITY_PATH, 
                            _MEDIUM_MODALITY_PATH
                        ]:
                            file_count = sum(1 for _ in category.glob("*.jpg"))
                        else:
                            file_count = sum(1 for _ in category.glob("*.txt"))
                        if file_count < _nb_catimgs:
                            logger.debug(f"[DEBUG]: missing stimuli in {path}")
                            missing_files = True
    
    if not missing_files:
        logger.debug("[DEBUG]: Data integrity verified!")
        return 
    
    logger.error("[ERROR]: Missing files detected")

    if _create_stimuli:
        logger.info("[INFO]: Recreating stimuli data...\n")

        if _STIMULI_PATH.exists(): shutil.rmtree(_STIMULI_PATH)
        stimgrab.create_stimuli(_nb_catimgs, _categories)

        logger.info("[INFO]: Stimuli data successfully generated!")
    else:
        raise FileNotFoundError("Missing stimuli data")

def get_stimuli() -> Dict[str, Path]:
    stimuli_dict = defaultdict(list)

    subject_modalities = [m for m in Modalities]
    n = _subjectid % NB_MODALITIES
    subject_modalities = subject_modalities[-n:] + subject_modalities[:-n]

    for img_path in _TARGET_PATH.rglob("*.jpg"):
        stimuli_dict[img_path.parent.name].append(
            (
                img_path, 
                subject_modalities[len(stimuli_dict[img_path.parent.name])%NB_MODALITIES]
            )
        )

    # flatten result to list
    stimuli_list = list(chain.from_iterable(stimuli_dict.values()))

    return stimuli_list

def get_modality_path(target_path:Path, modality:Modalities) -> Path:
    if modality == Modalities.HIGH:
        return _HIGH_MODALITY_PATH / "/".join(target_path.parts[-3:])
    elif modality == Modalities.MEDIUM:
        return _MEDIUM_MODALITY_PATH / "/".join(target_path.parts[-3:])
    elif modality == Modalities.MEDIUMLOW:
        if _language == Languages.EN:
            return Path(
                f"{_MEDIUMLOW_MODALITY_PATH}/{'/'.join(target_path.parts[-3:])}"
            ).with_name(Path(target_path.stem + "_en.txt").name)
        elif _language == Languages.FR:
            return Path(
                f"{_MEDIUMLOW_MODALITY_PATH}/{'/'.join(target_path.parts[-3:])}"
            ).with_name(Path(target_path.stem + "_fr.txt").name)
        else:
            return None
    elif modality == Modalities.LOW:
        if _language == Languages.EN:
            return Path(
                f"{_LOW_MODALITY_PATH}/{'/'.join(target_path.parts[-3:])}"
            ).with_name(Path(target_path.stem + "_en.txt").name)
        elif _language == Languages.FR:
            return Path(
                f"{_LOW_MODALITY_PATH}/{'/'.join(target_path.parts[-3:])}"
            ).with_name(Path(target_path.stem + "_fr.txt").name)
        else:
            return None
    else: 
        return None
    

# VISUAL =======================================================================

def get_fixation_cross(win) -> visual.ShapeStim:
    return visual.ShapeStim(
        win, vertices='cross', 
        size=(win.size[0]*.01, win.size[1]*.01*(win.size[0]/win.size[1])), 
        fillColor='black', lineColor=None)

def get_stimulus_frame(win) -> visual.Circle:
    return visual.Circle(win, size=win.size[0]*.5, color="gray")

def get_stimulus_aperture(win) -> visual.Aperture:
    aperture = visual.Aperture(win, size=win.size[0]*.4, shape='circle') 
    aperture.enabled = False
    return aperture

def get_mondrian_pattern(
        win,
        nb_rectangles, 
        pattern_scale=(1,1), 
        rect_scale=0.1,
        color=None,
        max_opacity=None) -> visual.ElementArrayStim:
    
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

def show_instructions(win:visual.Window):
    instruction_text = visual.TextStim(
        win,
        height=win.size[0]*.014,
        color="white",
        text=_TRIAL_INSTRUCTION[_language],
        wrapWidth=win.size[0]*.7
    )
    
    frame = visual.Rect(
        win, 
        fillColor=None, lineColor='white' ,
        size=(win.size[0]*.85, win.size[1]*.85)
    )
    
    instruction_text.draw()
    frame.draw()

    win.flip()

    event.waitKeys(keyList=['space'])

def run_delay(win:visual.Window, duration:float, 
              background:list=[], foreground:list=[]
):
    # draw background
    for stim in background: stim.draw()
    # draw foreground
    for stim in foreground: stim.draw()
    # update screen
    win.flip()
    # delay for duration seconds
    core.wait(duration)

def run_cue_presentation(win:visual.Window, cue_path:Path, modality:Modalities, 
                         duration:float, aperture:visual.Aperture, 
                         background:list=[], foreground:list=[]
):
    # draw background
    for stim in background: stim.draw()

    cue = None

    if modality in [Modalities.HIGH, Modalities.MEDIUM]:
        cue = visual.ImageStim(win, cue_path, size=win.size[0]*.4)
    elif modality in [Modalities.MEDIUMLOW, Modalities.LOW]:
        with open(cue_path, "r", encoding="utf-8") as cue_file:
            cue_text = cue_file.read()
        cue = visual.TextStim(
            win, 
            bold=True,
            height=win.size[0]*.02,
            color="white", 
            text=cue_text,
            wrapWidth=win.size[0]*.35
        )
    else:
        cue = visual.TextStim(
            win, 
            bold=True,
            height=win.size[0]*.02,
            color="white", 
            text="CONTROL",
            wrapWidth=win.size[0]*.35
        )
    
    if cue is not None:
        aperture.enabled = True
        cue.draw()
        aperture.enabled = False

    # draw foreground
    for stim in foreground: stim.draw()
    # update screen
    win.flip()
    # delay for duration seconds
    core.wait(duration)



def run_stimulus(
        win:visual.Window, target_path:Path, duration:float, 
        flash_interval:float, aperture:visual.Aperture, 
        background:list=[], foreground:list=[]
):
    response_time = None
    
    opacity = 1

    clock = core.Clock()
    event.clearEvents(eventType='keyboard')  # clear past keypresses

    start_time = clock.getTime()

    while clock.getTime() - start_time < duration:
        current_time = clock.getTime()

        # update opacity (linearly decreases over time)
        opacity = (current_time / duration)

        # draw background
        for stim in background: stim.draw()
    
        # draw CFS stimulus inside circular frame
        aperture.enabled = True

        target = visual.ImageStim(
            win=win,
            image=target_path,
            size=win.size[0]*.4
        )
        target.setColor(_red_filter)
        target.opacity = opacity
        target.draw()
        
        mask = get_mondrian_pattern(win, 100, (.7,.7), color=_cyan_filter, max_opacity=1-opacity)
        mask.draw()
        
        aperture.enabled = False

        # draw foreground
        for stim in foreground: stim.draw()
        # update screen
        win.flip()

        # check for key press during the loop
        keys = event.getKeys(timeStamped=clock)  # Capture key presses with timestamps
        if keys:
            response_time = keys[0][1]  # Capture the time at which the key was pressed
            print(f"Response Time: {response_time:.4f} seconds.")
            break
        
        # compute wait time based on time since last frame update
        wait_time = max(0, flash_interval - (clock.getTime() - current_time))
        core.wait(wait_time)
    
    if response_time is None: print(f"No response.")
    return response_time


def run_trial():
    win = visual.Window(monitor="testMonitor",  units="pix", fullscr=not _windowed, allowStencil=True)
    
    if _use_seed:
        np.random.seed(_random_seed)
    else:
        np.random.seed(int(time()))
    
    fixation_duration = .5 #s
    cue_duration = 3 #s
    delay_duration = .5 #s
    stimulus_duration = 6 #s

    flash_rate = 3  # Hz
    flash_interval = 1.0 / flash_rate  # s

    mondrian_background = get_mondrian_pattern(win, 200)
    stimulus_frame = visual.Circle(win, size=win.size[0]*.5, color="gray")  
    fixation_cross = get_fixation_cross(win)
    stimulus_aperture = get_stimulus_aperture(win)

    stimuli = get_stimuli()
    np.random.shuffle(stimuli)

    trial_log = SubjectLog()

    show_instructions(win)

    # run a cycle for each stimulus
    for stim in stimuli:
        # run fixation phase
        run_delay(
            win, fixation_duration, 
            background=[mondrian_background, stimulus_frame], 
            foreground=[fixation_cross]
        )
        # run cue presentation phase
        run_cue_presentation(win, get_modality_path(stim[0], stim[1]), stim[1], 
                             cue_duration, stimulus_aperture, 
                             background=[mondrian_background, stimulus_frame], 
                             foreground=[]
        )
        # run delay phase
        run_delay(
            win, delay_duration, 
            background=[mondrian_background, stimulus_frame], 
            foreground=[fixation_cross]
        )
        # run stimulus presentation phase
        reaction_time = run_stimulus(win, stim[0], stimulus_duration, 
                                     flash_interval, stimulus_aperture, 
                                     background=[mondrian_background, stimulus_frame], 
                                     foreground=[fixation_cross]
        )

        trial_log.log(stim[0], stim[1], reaction_time)

    _update_cache(_CacheKeys.SUBJECTS, _subjectid+1)

    trial_log.close()

    win.close()
    core.quit()

# MAIN EXECUTION ===============================================================

def main():
    _handle_args()
    _load_cache()
    _verify_files()

    run_trial()


if __name__ == "__main__":
    main()
    
    