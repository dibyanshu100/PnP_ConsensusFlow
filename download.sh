FILE=$1

if [ $FILE == "pretrained-network-celeba" ]; then
    OUTPUT_DIR=./model/celeba/ot
    mkdir -p $OUTPUT_DIR
    gdown 1-HcmIbwlFaJmGcLI5eKpDJy7EivIUQNe -O $OUTPUT_DIR/model_final.pt

elif  [ $FILE == "pretrained-network-afhq-cat" ]; then
    OUTPUT_DIR=./model/afhq_cat/ot
    mkdir -p $OUTPUT_DIR
    gdown 1OEcbfN2QNSjQVGeUIdR4YkQ1m3_B7jr6 -O $OUTPUT_DIR/model_final.pt

elif [ "$FILE" == "afhq-cat-dataset" ]; then
    DEST_DIR=./data/afhq_cat
    ZIP_FILE=./data/afhq-cat.zip
    FILE_ID=1dvbCSrMG9i_1YiPrxp4Dpw3rmqXPCgCj
    mkdir -p ./data
    mkdir -p "$DEST_DIR"
    echo "Downloading afhq-cat dataset from Google Drive..."
    gdown "$FILE_ID" -O "$ZIP_FILE"
    echo "Extracting..."
    unzip "$ZIP_FILE" -d "$DEST_DIR" -x "__MACOSX/*"
    echo "Removing macOS metadata..."
    rm -rf "$DEST_DIR/__MACOSX"
    if [ -d "$DEST_DIR/afhq_cat" ]; then
        echo "Flattening nested folder..."
        mv "$DEST_DIR/afhq_cat"/* "$DEST_DIR/"
        rm -rf "$DEST_DIR/afhq_cat"
    fi
    echo "Cleaning up..."
    rm -f "$ZIP_FILE"
    echo "Done!"


elif [ "$FILE" == "celeba-dataset" ]; then
    DEST_DIR=./data/celeba
    ZIP_FILE="$DEST_DIR/celeba.zip"
    EXTRACT_DIR="$DEST_DIR/extracted"
    IMG_DIR="$DEST_DIR/img_align_celeba"
    PART_FILE="$DEST_DIR/list_eval_partition.csv"
    FILE_ID=1cQlv1ebm9k3exfCXpTYhtUpDxB1pJKRb

    mkdir -p "$DEST_DIR" "$EXTRACT_DIR" "$IMG_DIR"
    mkdir -p "$DEST_DIR/train" "$DEST_DIR/val" "$DEST_DIR/test"

    echo "Downloading dataset from Google Drive..."
    if ! gdown "$FILE_ID" -O "$ZIP_FILE"; then
        echo "Error: download failed."
        exit 1
    fi

    if [ ! -f "$ZIP_FILE" ]; then
        echo "Error: ZIP file not found after download."
        exit 1
    fi

    echo "Extracting dataset..."
    unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR"
    rm "$ZIP_FILE"

    echo "Finding image files..."
    find "$EXTRACT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort > "$DEST_DIR/all_images.txt"

    if [ ! -s "$DEST_DIR/all_images.txt" ]; then
        echo "Error: no images found after extraction."
        exit 1
    fi

    total_images=$(wc -l < "$DEST_DIR/all_images.txt")
    echo "Found $total_images image files."

    echo "Moving all images into $IMG_DIR ..."
    moved=0
    while IFS= read -r img; do
        base=$(basename "$img")
        mv "$img" "$IMG_DIR/$base"
        moved=$((moved + 1))

        if [ $((moved % 5000)) -eq 0 ] || [ "$moved" -eq "$total_images" ]; then
            percent=$((moved * 100 / total_images))
            echo "Moved $moved / $total_images files (${percent}%)"
        fi
    done < "$DEST_DIR/all_images.txt"

    rm "$DEST_DIR/all_images.txt"
    rm -rf "$EXTRACT_DIR"

    echo "Creating train/val/test split and CSV partition file..."

    ls "$IMG_DIR" | sort > "$DEST_DIR/filelist.txt"
    total=$(wc -l < "$DEST_DIR/filelist.txt")

    train_count=$((total * 80 / 100))
    val_count=$((total * 10 / 100))
    test_count=$((total - train_count - val_count))

    # CSV header
    echo "image,split" > "$PART_FILE"

    c=0
    while IFS= read -r fname; do
        if [ "$c" -lt "$train_count" ]; then
            split=0
            cp -s "$IMG_DIR/$fname" "$DEST_DIR/train/$fname" 2>/dev/null || \
            ln -s "$(realpath "$IMG_DIR/$fname")" "$DEST_DIR/train/$fname"
        elif [ "$c" -lt $((train_count + val_count)) ]; then
            split=1
            cp -s "$IMG_DIR/$fname" "$DEST_DIR/val/$fname" 2>/dev/null || \
            ln -s "$(realpath "$IMG_DIR/$fname")" "$DEST_DIR/val/$fname"
        else
            split=2
            cp -s "$IMG_DIR/$fname" "$DEST_DIR/test/$fname" 2>/dev/null || \
            ln -s "$(realpath "$IMG_DIR/$fname")" "$DEST_DIR/test/$fname"
        fi

        # Write CSV line
        echo "$fname,$split" >> "$PART_FILE"

        c=$((c + 1))

        if [ $((c % 5000)) -eq 0 ] || [ "$c" -eq "$total" ]; then
            percent=$((c * 100 / total))
            echo "Processed $c / $total files (${percent}%)"
        fi
    done < "$DEST_DIR/filelist.txt"

    rm "$DEST_DIR/filelist.txt"

    echo "Done."
    echo "Images stored in: $IMG_DIR"
    echo "Split folders created:"
    echo "  $DEST_DIR/train"
    echo "  $DEST_DIR/val"
    echo "  $DEST_DIR/test"
    echo "Partition file created: $PART_FILE"

elif [ "$FILE" == "celebahq-dataset" ]; then
    if ! command -v kaggle &> /dev/null; then
        echo "Error: Kaggle API not found."
        echo "Install it with:"
        echo "    pip install kaggle"
        exit 1
    fi
 
    DEST_DIR=./data/celebahq
    RAW_DIR="$DEST_DIR/raw"
    IMG_DIR="$DEST_DIR/test"
    MASK_DIR="$DEST_DIR/mask_anno"
 
    echo "Creating dataset directory..."
    mkdir -p "$RAW_DIR" "$IMG_DIR" "$MASK_DIR"
 
    echo "Downloading CelebAMask-HQ from Kaggle..."
    kaggle datasets download \
        -d ipythonx/celebamaskhq \
        -p "$RAW_DIR" \
        --unzip
 
    echo ""
    echo "CelebAMask-HQ download complete."
    echo "Directory structure:"
    find "$RAW_DIR" -maxdepth 3 -type d
 
    echo ""
    echo "Locating the face-image folder..."
    SRC_IMG_DIR=$(find "$RAW_DIR" -type d -iname "CelebA-HQ-img" | head -n 1)
 
    if [ -z "$SRC_IMG_DIR" ]; then
        echo "Error: could not find a 'CelebA-HQ-img' folder inside the"
        echo "downloaded dataset (see the directory tree above). Move the"
        echo "face images (named like 0.jpg ... 29999.jpg) into $IMG_DIR"
        echo "yourself, then rerun with this dataset already present."
        exit 1
    fi
 
    echo "Found images at: $SRC_IMG_DIR"
    echo "Moving images into $IMG_DIR ..."
    mv "$SRC_IMG_DIR"/* "$IMG_DIR/"
 
    echo ""
    echo "Locating the ground-truth mask-annotation folder..."
    SRC_MASK_DIR=$(find "$RAW_DIR" -type d -iname "CelebAMask-HQ-mask-anno" | head -n 1)
 
    if [ -z "$SRC_MASK_DIR" ]; then
        echo "Warning: could not find a 'CelebAMask-HQ-mask-anno' folder"
        echo "inside the downloaded dataset (see the directory tree above)."
        echo "Only face images were saved -- ground-truth masks won't be"
        echo "available. Inspect $RAW_DIR manually if you need them."
    else
        echo "Found mask annotations at: $SRC_MASK_DIR"
        echo "Moving mask annotations into $MASK_DIR ..."
        mv "$SRC_MASK_DIR"/* "$MASK_DIR/"
    fi
 
    echo "Cleaning up..."
    rm -rf "$RAW_DIR"
 
    echo "Done!"
    echo "Images stored in: $IMG_DIR"
    echo "Mask annotations stored in: $MASK_DIR (if found -- see above)"
    echo ""
    echo "To use train/test splits, place train_ids.txt/test_ids.txt at"
    echo "$DEST_DIR/splits/ and pass their paths via --opts"
    echo "celebahq_train_ids <path> celebahq_test_ids <path>"
 
else
    echo "Available arguments are pretrained-network-celeba, pretrained-network-afhq-cat, celeba-dataset, afhq-cat-dataset and celebahq-dataset."
    exit 1
fi