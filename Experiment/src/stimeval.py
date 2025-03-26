print("importing modules...")
import time
start = time.time()

import pandas as pd
import utils.image_processing as imp

from pathlib import Path
from utils.clip_similarity import compare_images, compare_image_text

print(f"modules imported in {time.time() - start} seconds")


STIMULI_PATH = Path("../res/stimuli")
ORIGIN_PATH = STIMULI_PATH / "original" 
GRAYSCALE_PATH = STIMULI_PATH / "grayscale"
HICONGR_PATH = STIMULI_PATH / "trialready/high"
MIDCONGR_PATH = STIMULI_PATH / "trialready/medium"

METRICS_PATH = Path("../res/metrics.csv")

def image_metric(clipsim:float, ssim:float, alpha=.7):
    return alpha * clipsim + (1-alpha) * ssim

if __name__ == "__main__":
    metrics = []

    for img_path in GRAYSCALE_PATH.rglob("*.jpg"):
        img_name = img_path.as_posix()[len(GRAYSCALE_PATH.as_posix()):]

        print(f"Computring metrics for \"{img_name}\"")

        src_img = imp.read_image(img_path, imp.IOFlags.GRAYSCALE)
        hc_img = imp.read_image(f"{HICONGR_PATH}{img_name}", imp.IOFlags.GRAYSCALE)
        mc_img = imp.read_image(f"{MIDCONGR_PATH}{img_name}", imp.IOFlags.GRAYSCALE)
        
        hc_clip = compare_images(src_img, hc_img)
        hc_ssim = imp.get_ssim(src_img, hc_img)
        #hc_val = image_metric(hc_clip, hc_ssim)

        mc_clip = compare_images(src_img, mc_img)
        mc_ssim = imp.get_ssim(src_img, mc_img) 
        #mc_val = image_metric(mc_clip, mc_ssim)

        metrics.append({
            "image"     : img_name,
            "high_clip" : hc_clip,
            "high_ssim" : hc_ssim,
            "medium_clip"   : mc_clip,
            "medium_ssim"   : mc_ssim
        })
    
    df = pd.DataFrame(metrics)
    df.to_csv(METRICS_PATH, index=False)
    print()
    print("Results saved to \"metrics.csv\"")

    print()
    print()


        
        