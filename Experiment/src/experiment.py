import logging
import numpy as np
from pathlib import Path
from psychopy import visual, core, event

# CONSTANTS ====================================================================

WIN = visual.Window(monitor="testMonitor",  units="pix", fullscr=True)

TRIAL_INSTRUCTION = "" \
"Please press the space bar as soon as you see an image appear trough the mask."

# CONFIG =======================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# UTILITY ======================================================================

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
        text=TRIAL_INSTRUCTION
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
    show_instructions()
    background = get_mondrian_pattern(200)
    while True:
        background.draw()
        get_fixation_cross().draw()
        WIN.flip()
        event.waitKeys(keyList=['space'])
        break
    WIN.close()
    core.quit()