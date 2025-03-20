#print("importing modules...")
#import time
#start = time.time()

import utils.image_processing as imp
from pathlib import Path

#print(f"modules imported in {time.time() - start} seconds")

DFDIM = (224,224)

STIMULI_PATH = Path("../res/stimuli")
ORIGIN_PATH = STIMULI_PATH / "original" 
GRAYSCALE_PATH = STIMULI_PATH / "grayscale"
HICONGR_PATH = STIMULI_PATH / "trialready/high"
MIDCONGR_PATH = STIMULI_PATH / "trialready/medium"


def medium_congruence(img, img_path):
    mc_img = imp.lowpass_filter(img,50)
    
    dest = Path(
        f"{HICONGR_PATH}{img_path.as_posix()[len(ORIGIN_PATH.as_posix()):]}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
        
    imp.save_image(dest, mc_img)

def high_congruence(img, img_path):
    hc_img = imp.xdog(img, sigma=0.5, k=100, gamma=.7, epsilon=0.3, phi=1)
    hc_img = imp.otsu_thresholding(hc_img)
    
    dest = Path(
        f"{MIDCONGR_PATH}{img_path.as_posix()[len(ORIGIN_PATH.as_posix()):]}"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    imp.save_image(dest, hc_img)

if __name__ == "__main__":
    for img_path in ORIGIN_PATH.rglob("*.jpg"):
        print(f"Treating \"{img_path.as_posix()[len(ORIGIN_PATH.as_posix()):]}\"")

        img = imp.read_image(img_path, flag=imp.IOFlags.GRAYSCALE, dim=DFDIM)
        
        print("\t> Ensuring grayscale...")
        img = imp.ensure_grayscale(img)
        
        dest = Path(
            f"{GRAYSCALE_PATH}{img_path.as_posix()[len(ORIGIN_PATH.as_posix()):]}"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        imp.save_image(dest, img)

        print("\t> Converting using medium congruence method...")
        high_congruence(img, img_path)
        
        print("\t> Converting using high congruence method...")
        medium_congruence(img, img_path)



    
