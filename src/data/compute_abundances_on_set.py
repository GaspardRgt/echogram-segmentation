
import os

import argparse
import numpy as np

"""
Computes classes abundances in a given dataset.
"""

parser = argparse.ArgumentParser()

parser.add_argument("--mask_path",
                    type=str)

args = parser.parse_args()

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def count_classes(mask_path, num_classes=9):
    os.chdir(mask_path)

    classes = [i for i in range(num_classes)]
    class_abundances = np.array([0]*num_classes)

    for path in listdir_nohidden(mask_path):
        os.chdir(mask_path)
        
        y = np.load(path)
        
        for i in range(num_classes):
            class_abundances[i] += np.count_nonzero(y == i)

    print(f"Classes: {classes}\nClasses abundance: {class_abundances}")
    class_freq = class_abundances / class_abundances.sum()
    
    class_abundances[0] = 0
    class_freq_without_0 = class_abundances / class_abundances.sum()

    class_abundances[2] = 0
    class_freq_without_0_and_2 = class_abundances / class_abundances.sum()
    
    print(f'\nClass frequencies: {class_freq}')
    print(f'\nClass frequencies without 0: {class_freq_without_0}')
    print(f'\nClass frequencies without 0 and 2: {class_freq_without_0_and_2}')
    
def main():
    count_classes(mask_path=args.mask_path,
                  num_classes=9)

if __name__ == "__main__":
    main()