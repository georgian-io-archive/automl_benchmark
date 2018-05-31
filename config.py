#!/usr/bin/env python

import json
import sys

#Loads configuration to file. Provides direct python access.
def load_config():

    #Load Config
    with open('batch.config') as f:
        data = json.load(f)

    return data


#This function is used to provide configuration to shell scripts
def print_config():

    data = load_config()

    #Export config value to stdin
    if len(sys.argv) > 1:
        print(data[sys.argv[1]])
    else:
        print("None")


if __name__ == '__main__':
    print_config()