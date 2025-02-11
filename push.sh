#!/bin/bash

# Check if a filename was provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a filename"
    echo "Usage: ./push.sh <filename>"
    exit 1
fi

FILENAME=$1

# Check if the file exists
if [ ! -f "$FILENAME" ]; then
    echo "Error: $FILENAME not found"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "Error: not a git repository"
    exit 1
fi

# Add the file
git add "$FILENAME"

# Get the current branch name
BRANCH=$(git symbolic-ref --short HEAD)

# Commit with a message
echo "Committing to branch: $BRANCH"
git commit -m "Update $FILENAME"

# Push to the remote repository
echo "Pushing to remote..."
git push origin $BRANCH

if [ $? -eq 0 ]; then
    echo "Successfully pushed to GitHub"
else
    echo "Error: Push failed"
    exit 1
fi