print("importing modules...")
import time
start = time.time()

import pandas as pd
import utils.image_processing as imp

from pathlib import Path
from utils.clip_similarity import compare_images, compare_image_text, init_clip

print(f"modules imported in {time.time() - start} seconds")

_RELATIVE_PATH = Path(__file__).resolve().parent

STIMULI_PATH = (_RELATIVE_PATH / "../res/stimuli").resolve()
ORIGIN_PATH = STIMULI_PATH / "original" 
GRAYSCALE_PATH = STIMULI_PATH / "grayscale"
HICONGR_PATH = STIMULI_PATH / "trialready/high"
MIDCONGR_PATH = STIMULI_PATH / "trialready/medium"
MIDLOWCONGR_PATH = STIMULI_PATH / "trialready/mediumlow"
LOWCONGR_PATH = STIMULI_PATH / "trialready/low"

METRICS_PATH = (_RELATIVE_PATH / "../res/expdata/metrics.csv").resolve()

def image_metric(clipsim:float, ssim:float, alpha=.7):
    return alpha * clipsim + (1-alpha) * ssim

if __name__ == "__main__":
    init_clip()

    metrics = []

    for img_path in GRAYSCALE_PATH.rglob("*.jpg"):
        img_name = img_path.as_posix()[len(GRAYSCALE_PATH.as_posix()):]

        print(f"Computring metrics for \"{img_name}\"")

        src_img = imp.read_image(img_path, imp.IOFlags.GRAYSCALE)
        
        hc_img = imp.read_image(f"{HICONGR_PATH}{img_name}", imp.IOFlags.GRAYSCALE)
        
        mc_img = imp.read_image(f"{MIDCONGR_PATH}{img_name}", imp.IOFlags.GRAYSCALE)
        
        mlc_txt = ""
        mlc_file = Path(f"{MIDLOWCONGR_PATH}{img_name}")
        mlc_file = mlc_file.with_name(mlc_file.stem + "_en.txt")
        with open(mlc_file, 'r', encoding='utf-8') as f:
            mlc_txt = f.read()
        
        lc_txt = ""
        lc_file = Path(f"{LOWCONGR_PATH}{img_name}")
        lc_file = lc_file.with_name(lc_file.stem + "_en.txt")
        with open(lc_file, 'r', encoding='utf-8') as f:
            lc_txt = f.read()

        hc_clip = compare_images(src_img, hc_img)
        hc_ssim = imp.get_ssim(src_img, hc_img)

        mc_clip = compare_images(src_img, mc_img)
        mc_ssim = imp.get_ssim(src_img, mc_img) 
        
        mlc_clip = compare_image_text(src_img, mlc_txt)

        lc_clip = compare_image_text(src_img, lc_txt)

        metrics.append({
            "image"             : img_name,
            "high_clip"         : hc_clip,
            "high_ssim"         : hc_ssim,
            "medium_clip"       : mc_clip,
            "medium_ssim"       : mc_ssim,
            "medium_low_clip"   : mlc_clip,
            "low_clip"          : lc_clip
        })
    
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics)
    df.to_csv(METRICS_PATH, index=False)
    
    print()
    print(f"Results saved to {METRICS_PATH}")

    print()
    print()


        
        