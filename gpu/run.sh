#!/bin/bash

#Check if an argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <file.npy>"
  exit 1
fi

# Validate the file extension
INPUT_FILE="$1"
if [[ "$INPUT_FILE" != *.npy ]]; then
  echo "Error: The input file must have a .npy extension."
  exit 1
fi

# Set the build directory
BUILD_DIR="build"

# Check if the build directory exists, if not create it
if [ ! -d "$BUILD_DIR" ]; then
  mkdir "$BUILD_DIR"
fi

# Run meson setup (configure build environment)
meson setup "$BUILD_DIR" || exit 1

# Compile the project
meson compile -C "$BUILD_DIR" || exit 1

# Find the executable (assuming only one main executable)
EXECUTABLE=$(find "$BUILD_DIR" -type f -executable -print -quit)

# Run the executable if it exists
if [ -x "$EXECUTABLE" ]; then
  echo "Running the project:"
  "$EXECUTABLE" $INPUT_FILE
else
  echo "Executable not found in $BUILD_DIR"
fi