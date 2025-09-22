#!/bin/bash
pip3 install jupyter_contrib_nbextensions notebook==6.4.12
pip3 install notebook==6.4.12

for f in $(find ../tutorial_exercises -name "*.ipynb"); do
    if [[ $f == *"nbconvert"* ]]; then
        continue
    fi
    echo "Testing notebook: $f"
    # Capture the output and only print it if the notebook fails
    jupyter nbconvert --to notebook --execute $f > /tmp/nbtest.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Failed: $f"
        cat /tmp/nbtest.log
        echo "Failed: $f"
        find ../tutorial_exercises -name "*.nbconvert.ipynb" -exec rm {} \;
        exit 1
    fi
done
find ../tutorial_exercises -name "*.nbconvert.ipynb" -exec rm {} \;