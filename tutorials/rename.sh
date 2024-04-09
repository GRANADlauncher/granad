#!/bin/bash

# Iterate over all .py files in the directory
for filename in *.py; do
    echo "Processing $filename..."

    # Find the first line starting with "# ##"
    pattern_line=$(grep -m 1 '^# ##' "$filename")

    # Check if the pattern was found
    if [ -z "$pattern_line" ]; then
        echo "Pattern '# ##' not found in $filename."
        continue
    fi

    # Extract words following "# ##", convert them to lowercase, and replace spaces with underscores
    new_name=$(echo "$pattern_line" | sed 's/^# ## //' | tr '[:upper:]' '[:lower:]' | tr ' ' '_').py
    new_path="${new_name}"

    # Rename the file if the new name does not conflict with an existing file
    if [ -e "$new_path" ]; then
        echo "Error: $new_path already exists. Skipping $filename."
    else
        mv "$filename" "$new_path"
        echo "Renamed $filename to $new_path"
    fi
done
