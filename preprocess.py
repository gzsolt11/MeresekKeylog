# Call python file as
# python preprocess.py dataset processed_data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import re
import numpy as np
from sklearn import metrics
import os
import shutil
import hashlib
import argparse

KEYS = ["0","1","2","3","4","5","6","7","8","9"]

# Adding to files every txt file in the baseline folders inside the input file
def beolvas(read_path):
    files = []
    for i in range(0,113):
        folder = str(i)
        fn = pathlib.Path(pathlib.Path(read_path) / folder).rglob('*.csv')
        files = files + [x for x in fn]
    return files
    

def parse_data(read_path, output_path):
    files = beolvas(read_path)


    output = []
    output.append(("H1","H2","H3","H4","H5","H6","PP1","PP2","PP3","PP4","PP5","RP1","RP2","RP3","RP4","RP5","user_id"))

    
    for file in files:
        # Getting the user_id, session_id, keyboard_id, task_id from the name of the file
        user_id = str(file).split("\\")[-2]

    

        # Opening file
        with open(file, "r") as file0:

            # In this list we store every key with the press or release times
            key_with_times = []

            for line in file0:
                # splitting the lines
                key, press_time, release_time = line.split(",")

                # If the key is not in the list
                if key not in KEYS:
                    continue

                key_with_times.append([key, press_time, release_time])

            

            #print(len(key_with_times))
            i = 0
            while i < len(key_with_times):
                
                # Calculating the Holdtime1, holdtime2, PressPress, ReleasePress times from the rows
                H1 = int(key_with_times[i][2]) - int(key_with_times[i][1])
                H2 = int(key_with_times[i+1][2]) - int(key_with_times[i+1][1])
                H3 = int(key_with_times[i+2][2]) - int(key_with_times[i+2][1])
                H4 = int(key_with_times[i+3][2]) - int(key_with_times[i+3][1])
                H5 = int(key_with_times[i+4][2]) - int(key_with_times[i+4][1])
                H6 = int(key_with_times[i+5][2]) - int(key_with_times[i+5][1])

                PP1 = int(key_with_times[i+1][1]) - int(key_with_times[i][1])
                PP2 = int(key_with_times[i+2][1]) - int(key_with_times[i+1][1])
                PP3 = int(key_with_times[i+3][1]) - int(key_with_times[i+2][1])
                PP4 = int(key_with_times[i+4][1]) - int(key_with_times[i+3][1])
                PP5 = int(key_with_times[i+5][1]) - int(key_with_times[i+4][1])

                RP1 = int(key_with_times[i+1][1]) - int(key_with_times[i][2])
                RP2 = int(key_with_times[i+2][1]) - int(key_with_times[i+1][2])
                RP3 = int(key_with_times[i+3][1]) - int(key_with_times[i+2][2])
                RP4 = int(key_with_times[i+4][1]) - int(key_with_times[i+3][2])
                RP5 = int(key_with_times[i+5][1]) - int(key_with_times[i+4][2])

                # If the PP and RP looks accurate we add it to the output
                #if PP < 1000 and abs(RP) < 1000:
                output.append((H1,H2,H3,H4,H5,H6,PP1,PP2,PP3,PP4,PP5,RP1,RP2,RP3,RP4,RP5,user_id))
                
                i = i + 6

                
        
        
    # Write processed data to file
    write_file = output_path + "/" + "osszesitett.csv"
    try:
        os.makedirs(write_file[:-17])
    except:
        pass
    try:
        with open(write_file, "a") as file:
            for entry in output:
                file.write(str(entry[0]) + "," + str(entry[1]) + "," + str(entry[2]) + "," + str(entry[3]) + "," + str(entry[4])+","+ str(entry[5]) + "," + str(entry[6]) + "," + str(entry[7]) + "," + str(entry[8])+ "," + str(entry[9])+","+ str(entry[10]) + "," + str(entry[11]) + "," + str(entry[12]) + "," + str(entry[13])+ ","+str(entry[14]) + "," + str(entry[15]) + "," + str(entry[16])  + "\n")
            file.close()
    except:
        with open(write_file, "w+") as file:
            for entry in output:
                file.write(str(entry[0]) + "," + str(entry[1]) + "," + str(entry[2]) + "," + str(entry[3]) + "," + str(entry[4])+","+ str(entry[5]) + "," + str(entry[6]) + "," + str(entry[7]) + "," + str(entry[8])+ "," + str(entry[9])+","+ str(entry[10]) + "," + str(entry[11]) + "," + str(entry[12]) + "," + str(entry[13])+ ","+str(entry[14]) + "," + str(entry[15]) + "," + str(entry[16])  + "\n")
            file.close()

            
        
def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return (i, x.index(v))
       
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_path", metavar="INPUT_PATH", help="Path to read raw typing data from.")
    parser.add_argument(dest="output_path", metavar="OUTPUT_PATH", help="Path to write processed data to")

    args = parser.parse_args()

    # Verify that input path exists
    assert os.path.exists(args.input_path), "Specified input path does not exist."

    # Check if path for preprocessed data exists
    if os.path.exists(args.output_path):
        ans = input("All preprocessed data will be overwritten. Do you want to continue? (Y/n) >> ")
        if not(ans == "" or ans.lower() == "y" or ans.lower() == "yes"):
            exit()

    # Creates fresh path for the preprocessed data
    if os.path.exists(args.output_path):
        if "processed_data" not in args.output_path:
            print("Processed data path must include \"processed_data\" as a precaution.")
        else:
            shutil.rmtree(args.output_path)
    os.mkdir(args.output_path)

    # Process the data
    parse_data(args.input_path, args.output_path)

    print("Data was preprocessed successfully.")


if __name__ == "__main__":
    main()

