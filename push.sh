#!/bin/bash

# Check if any files were provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide at least one filename"
    echo "Usage: ./push.sh <file1> [file2] [file3] ..."
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "Error: not a git repository"
    exit 1
fi

# Check if all files exist before proceeding
for FILE in "$@"; do
    if [ ! -f "$FILE" ]; then
        echo "Error: $FILE not found"
        exit 1
    fi
done

# Add all specified files
echo "Adding files..."
for FILE in "$@"; do
    echo "  Adding $FILE"
    git add "$FILE"
done

# Get the current branch name
BRANCH=$(git symbolic-ref --short HEAD)

# Create commit message based on number of files
if [ $# -eq 1 ]; then
    COMMIT_MSG="Update $1"
else
    COMMIT_MSG="Update multiple files: $*"
fi

# Commit with a message
echo "Committing to branch: $BRANCH"
git commit -m "$COMMIT_MSG"

# Push to the remote repository
echo "Pushing to remote..."
git push origin $BRANCH

if [ $? -eq 0 ]; then
    echo "Successfully pushed to GitHub"
else
    echo "Error: Push failed"
    exit 1
fi