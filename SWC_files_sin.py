#=======================================================================================================================================================================
#          Manipulating SWC files (one file at a time)
#=======================================================================================================================================================================
from ast import main
from cloudvolume import CloudVolume
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg') # Use 'Agg' backend for headless environments (no display)
import vtk
from mpl_toolkits.mplot3d import Axes3D
import neuron 
from neuron import h, gui
from neuron.units import ms, mV, um
import os # os stands for operating system, and it's a built-in Python module that provides a way to interact with the underlying operating system. It allows you to perform various tasks such as file and directory manipulation, environment variable access, and more. 
import pprint # pprint stands for "pretty-print" and is a built-in Python module that provides a way to print data structures in a more readable and organized format.
import sys  # sys is a built-in Python module that provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter. It allows you to manipulate the Python runtime environment, access command-line arguments, and perform various system-related tasks.

def enablePrint():
    sys.stdout = sys.__stdout__

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

print(f'VTK version: {vtk.VTK_VERSION}')

try:
    import cloudvolume
    print("CloudVolume is installed.")
except ImportError:
    print("CloudVolume is not installed. Please install it using 'pip install cloud-volume'.")
    # You can also add code here to exit the script or attempt installation
    # import sys
    # sys.exit(1) 
    
h.load_file('import3d.hoc')

# code to source only 1 SWC file.
swc_path = '/home/aksay_lab/NeuronProject/CloudvolNeuron_Test/SWC_files/76182_reRoot_reSample_5000.swc'  
print("File exists?", os.path.exists(swc_path))
print("Loading SWC file:", swc_path)

reader = h.Import3d_SWC_read() # This line creates an instance of the Import3d_SWC_read class, which is a built-in class in NEURON that is used to read SWC files. The reader object will be used to load the SWC file and extract the morphological data to create a cell model in NEURON.
reader.input(swc_path) # This line calls the input() method of the reader object, passing the path to the SWC file as an argument. The input() method reads the SWC file and prepares the data for instantiation. It processes the morphological information contained in the SWC file, such as the coordinates of the points, their types, and their connectivity, so that it can be used to create a cell model in NEURON.

importer = h.Import3d_GUI(reader, 0)
# Create a plain Python object to hold sections
class Cell:
    def __init__(self):
        self.sl = h.SectionList()   # real HOC SectionList

cell = Cell()

importer.instantiate(cell) # This line calls the instantiate() method of the importer object, passing the cell object as an argument. The instantiate() method takes the morphological data that was read from the SWC file and uses it to create a cell model in NEURON. It generates the sections and their connectivity based on the information in the SWC file and populates the cell.sl SectionList with the created sections. After this line is executed, the cell object will contain a SectionList that represents the morphology of the neuron as defined in the SWC file.
sl = h.SectionList()
shape = h.Shape(sl)
shape.exec_menu('3D Rotate')
#shape.show(0)
#shape.flush()

input("Press Enter to exit...")

