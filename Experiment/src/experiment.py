from pathlib import Path


# select images
#   3 per category
#   
RELATIVE_PATH = Path(__file__).parent
STIMULI_ORIGINAL_PATH = RELATIVE_PATH / "../res/stimuli/original"



def select_original():
    if not STIMULI_ORIGINAL_PATH.exists():
        raise FileNotFoundError("Original stimuli folder does not exist!")
    
    super_categories = list(STIMULI_ORIGINAL_PATH.iterdir())
    
    if not super_categories:
        raise FileNotFoundError("Missing original stimuli files!")
    
    categories = []
    for spcat in super_categories:
        cat_spcat += list(Path(spcat).iterdir())
        

    

if __name__ == "__main__":
    select_original()