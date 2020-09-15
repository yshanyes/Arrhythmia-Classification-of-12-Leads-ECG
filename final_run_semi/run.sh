#!/bin/bash
#
# file: run.sh
# In order to reduce the workload of testing algorithms, this bash script offers an example
# to call models to generate predictions on a certain folder with data (mat files),
# by calling an example script "challenge.py", to generate "answers.csv".
# All teams should modify this file with all the command lines needed, 
# and generate predictions with exact the same layout of "answers.csv"


echo "==== running entry script on the test set ===="

# Clear previous answers.csv, if needed
rm -f answers.csv

# Generate new answers.csv
python challenge.py --test_path /media/jdcloud1/Test/TEST8000

echo "=================== Done! ===================="


