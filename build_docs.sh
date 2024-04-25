#!/bin/bash

# Check if a command line argument (file name) was provided
if [ "$#" -eq 1 ]; then
    file_list="docs_src/tutorials/$1"
    single_file_mode=true
else
    file_list="docs_src/tutorials/*.py"
    single_file_mode=false
fi

# Directory setup for output
source_dir="docs_src/tutorials"
target_dir="docs/tutorials"
mkdir -p "$target_dir"

# Iterate over specified file(s) in the directory
for filename in $file_list; do
    echo "Processing $filename..."

    # If single file mode, convert this file to a notebook and markdown
    if [ "$single_file_mode" = true ]; then
        base_name=$(basename "$filename" .py)
        jupytext --to notebook --execute "$filename" -o "$target_dir/$base_name.ipynb"
        jupyter nbconvert --to markdown --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["remove_cell"]' "$target_dir/$base_name.ipynb"
    fi
done

# Convert index page and all tutorials only if no file argument is provided
if [ "$single_file_mode" = false ]; then
    jupytext --to notebook --execute docs_src/index.py -o docs/index.ipynb
    jupyter nbconvert --to markdown --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["remove_cell"]' docs/index.ipynb

    # Loop over all Python files in the source directory
    find "$source_dir" -name "*.py" -type f | while read py_file; do
        base_name=$(basename "$py_file" .py)
        jupytext --to notebook --execute "$py_file" -o "$target_dir/$base_name.ipynb"
        jupyter nbconvert --to markdown --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["remove_cell"]' "$target_dir/$base_name.ipynb"
    done
fi
