from pyedflib import highlevel
import pyedflib as plib
import numpy as np
import matplotlib.pyplot as plt
import src.dataParser




if __name__ == "__main__":
    parser = src.dataParser.DataParser()
    parser.load_dataset()
    parser.preprocess_data()
    
            
