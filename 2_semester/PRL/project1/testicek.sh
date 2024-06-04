#!/bin/bash



  # Step 1: Run generateNums.sh with given argument and save output to `implementation`
  sh run.sh "$1" > implementation

  # Step 2: Remove first line from file1 and save it into `input`
  head -n 1 implementation > input
  sed -i '1d' implementation

  # Step 3: Run python script and save output to `actual`
  python3 sort.py > actual

  # Step 4: Compare implementation and actual
  if diff -q implementation actual > /dev/null; then
      # Step 5: If there is no difference, write "same"
       echo "$1 same"
  else
      # Step 5: If there are differences, write "different"
      echo "$1 different"
  fi

