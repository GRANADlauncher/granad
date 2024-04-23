#!/bin/bash

# Iterate over all .py files in the directory
for filename in docs_src/tutorials/*.py; do
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
    new_path="docs_src/tutorials/${new_name}"

    # Rename the file if the new name does not conflict with an existing file
    if [ -e "$new_path" ]; then
        echo "Error: $new_path already exists. Skipping $filename."
    else
        mv "$filename" "$new_path"
        echo "Renamed $filename to $new_path"
    fi
done

# black src/granad/*; isort --profile=black src/granad/*
# black notebooks/tutorials/*; isort --profile=black notebooks/tutorials/*

# convert index page
jupytext --to notebook --execute docs_src/index.py -o docs/index.ipynb
jupyter nbconvert --to markdown --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["remove_cell"]' docs/index.ipynb

# convert tutorials
source_dir="docs_src/tutorials"
target_dir="docs/tutorials"
mkdir -p "$target_dir"

# Loop over all Python files in the source directory
find "$source_dir" -name "*.py" -type f | while read py_file; do
    # Get the base name without extension
    base_name=$(basename "$py_file" .py)
    
    # run notebooks
    jupytext --to notebook --execute "$py_file" -o "$target_dir/$base_name.ipynb"

    # convert to markdown
    jupyter nbconvert --to markdown --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["remove_cell"]' "$target_dir/$base_name.ipynb"

done
