#!/bin/bash

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# source zip file name
ZIP_FILE="$SRC_DIR/../res/datasets/mit_stimuli_scenes.zip"

# categories list to be extracted
FOLDER_LIST="$SRC_DIR/../res/categories.txt"

# loop through each line in the text file and extract category folder
while IFS= read -r folder; do
    if [[ -n "$folder" ]]; then 
        echo "Extracting $folder..."
        unzip -q -n "$ZIP_FILE" "$folder/*.jpg" -d "$SRC_DIR/../res/stimuli/original/"
    fi
done < "$FOLDER_LIST"

echo "Extraction completed."
