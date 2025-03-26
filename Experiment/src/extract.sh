#!/bin/bash

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DATASET_PATH="$SRC_DIR/../res/datasets/"
# source zip file name
DATASET_FILE="$DATASET_PATH/mit_stimuli_scenes.zip"

DATASET_URL="http://olivalab.mit.edu/MM/downloads/Scenes.zip"

# categories list to be extracted
FOLDER_LIST="$SRC_DIR/../res/categories.txt"

OUTPUT_PATH="$SRC_DIR/../res/stimuli/original/"

# Check if the categories list file exists
if [[ ! -f "$FOLDER_LIST" ]]; then
    echo "Error: Categories list '$FOLDER_LIST' not found!"
    exit 1
fi

# Check if the dataset zip file exists
if [[ ! -f "$DATASET_FILE" ]]; then
    echo -e "Dataset file not found! Downloading from $DATASET_URL...\n"
    
    # Ensure the dataset directory exists
    mkdir -p "$DATASET_PATH"
    
    # Download the zip file using wget
    wget -O "$DATASET_FILE" "$DATASET_URL"
    
    # Check if the download was successful
    if [[ ! -f "$DATASET_FILE" ]]; then
        echo "Error: Failed to download the dataset!"
        exit 2
    else
        echo "Dataset successfully downloaded!"
    fi
else
    echo "Dataset file found!"
fi

echo -e "\nExtracting files from category list...\n"

# loop through each line in the text file and extract category folder
while IFS= read -r folder; do
    if [[ -n "$folder" ]]; then 
        echo "Extracting $folder..."
        mkdir -p "$OUTPUT_PATH"
        unzip -q -n "$DATASET_FILE" "$folder/*.jpg" -d "$OUTPUT_PATH"
    fi
done < "$FOLDER_LIST"

echo -e "\nExtraction completed."