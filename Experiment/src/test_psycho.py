from psychopy import visual, core, event
import numpy as np

# Window setup for stereo (left-eye, right-eye)
win = visual.Window(size=(1024, 768), units="pix", fullscr=False, stereo=False)

# Parameters
flash_rate = 3  # Hz
flash_interval = 1.0 / flash_rate  # seconds
duration = 5  # seconds for the whole CFS trial
num_masks = int(duration * flash_rate)


def generate_background(win):
    back = visual.ShapeStim(win, vertices='rectangle', size=(win.size[0], win.size[1]), 
        fillColor='gray', lineColor=None)
    back.draw()

# Mondrian mask generator (random colored rectangles)
def generate_mondrian(win, max_opacity):
    # Create a random Mondrian pattern
    for _ in range(60):  # Random rectangles
        rect = visual.Rect(win, width=np.random.randint(50, 120), height=np.random.randint(70, 220),
                           pos=(np.random.randint(-256, 256), np.random.randint(-256, 256)),
                           fillColor=[-.1, -.7, -.7], lineColor=None)
        rect.opacity = np.random.uniform(0, max_opacity)
        rect.draw()


# Target stimulus (static image or shape)
target = visual.ImageStim(win, image='../res/stimuli/grayscale/64-scenes/train-68/3-1.jpg', pos=(0,0))
color_filter = np.array([-.75, -.1, -.1])
target.setColor(color_filter)
# Experiment loop
clock = core.Clock()
# Initial values for timing and opacity
opacity = 1
start_time = clock.getTime()
while clock.getTime() - start_time < duration:
    # Get the current time
    current_time = clock.getTime()

    # Update opacity (smoothly decreases over time)
    opacity = (current_time / duration)

    # Call your functions to generate background and mondrian mask
    generate_background(win)
    target.opacity = opacity
    target.draw()
    generate_mondrian(win, 1-opacity)

    # Update the display
    win.flip()

    # Control timing: Wait for the remaining time in the loop
    elapsed_time = clock.getTime() - current_time  # Time since last frame update
    core.wait(flash_interval - elapsed_time)

# Close the window
win.close()
core.quit()
