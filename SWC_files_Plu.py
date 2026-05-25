#=======================================================================================================================================================================
#          Manipulating SWC files for multiple files (needs revision)
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
from neuron import n
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
'''swc_path = '/home/jd/NeuronProject/CloudvolNeuron_Test/76182.swc'  # adjust!
print("File exists?", os.path.exists(swc_path))'''

swc_folder = '/home/jd/NeuronProject/CloudvolNeuron_Test/SWC_files/'
swc_files = [os.path.join(swc_folder, f) for f in os.listdir(swc_folder) if f.endswith('.swc')] #.join concatenates one or more paths. for f in os.listdir(swc_folder) iterates through all files in the specified folder, and if f.endswith('.swc') checks if the file has a .swc extension. If it does, the full path to the file is created by joining the folder path and the file name, and this path is added to the swc_files list. This results in a list of full paths to all SWC files in the specified folder.

print()
for count, swc_path in enumerate(swc_files, 1):
    print(f'File #{count} to: {swc_path} ')
print()

all_cells = [] # this list will hold all the cell objects created from the SWC files. 

for swc_path in swc_files:
    print("Loading SWC file:", swc_path)

    reader = h.Import3d_SWC_read() # This line creates an instance of the Import3d_SWC_read class, which is a built-in class in NEURON that is used to read SWC files. The reader object will be used to load the SWC file and extract the morphological data to create a cell model in NEURON.
    reader.input(swc_path) # This line calls the input() method of the reader object, passing the path to the SWC file as an argument. The input() method reads the SWC file and prepares the data for instantiation. It processes the morphological information contained in the SWC file, such as the coordinates of the points, their types, and their connectivity, so that it can be used to create a cell model in NEURON.

    importer = h.Import3d_GUI(reader, 0) # This line creates an instance of the Import3d_GUI class, which is a built-in class in NEURON that provides a graphical user interface for importing 3D morphological data. The importer object is initialized with the reader object (which contains the loaded SWC data) and a flag (0 in this case) that specifies whether to use the GUI or not. The importer will be used to instantiate the cell model based on the data read from the SWC file.
    
    # Create a plain Python object to hold sections
    class Cell:
        def __init__(self):
            self.sl = h.SectionList()   # real HOC SectionList

    cell = Cell()

    importer.instantiate(cell) # This line calls the instantiate() method of the importer object, passing the cell object as an argument. The instantiate() method takes the morphological data that was read from the SWC file and uses it to create a cell model in NEURON. It generates the sections and their connectivity based on the information in the SWC file and populates the cell.sl SectionList with the created sections. After this line is executed, the cell object will contain a SectionList that represents the morphology of the neuron as defined in the SWC file.

    # After instantiation, collect only this cell's sections
    for sec in h.allsec():
        if sec not in [s for c in all_cells for s in getattr(c, "sl", [])]:
            cell.sl.append(sec) # sl is the SectionList attribute of the Cell class, and sec is the current section being iterated over. The append() method is used to add the section to the SectionList of the current cell. This way, each cell object will have its own list of sections that belong to it, allowing for better organization and manipulation of the morphology data.

    all_cells.append(cell)

    # Neuron gui pops up with the line code: importer.instantiate(cell) and the script execution halts until you close the window. After you close the window, the script continues to execute and collects the sections for that cell before moving on to the next SWC file in the loop. This process repeats for each SWC file, allowing you to load and instantiate multiple neuron morphologies sequentially.


   
#---------------------------------------------------------------------------------------------------------------------------------------
'''print('\nWhat exists in the object Cell:\n')
pprint.pprint(dir(cell))
print()
# print("Total points:", cell.total_points) #doesn't work for some reason.

print("\nAvailable attributes and methods in the Cell object that do not have '_' separating them: \n") # the methods aare separa
for name in dir(cell):
    if not name.startswith('_'):
        print(name)

for sec in h.allsec(): # h.allsec() returns a generator that yields all sections in the model. By converting it to a list, we can easily count the number of sections and iterate through them.
    print(h.secname(sec=sec)) #By passing sec=sec, we specify which section's name we want to retrieve. This will print the name of each section in the model after instantiation.

h.topology() # this function prints the connectivity of the sections in the model, showing how they are connected to each other. It provides a hierarchical view of the sections and their relationships.

total_pts = 0
for sec in h.allsec():
    total_pts += int(h.n3d(sec=sec))

print("After instantiate - sections in model:", len(list(h.allsec()))) # this will print the total number of sections in the model after instantiation. By converting h.allsec() to a list, we can easily count the number of sections and print it out.
print("Total 3D points in model:", total_pts) # points are the coordinates that define the morphology of the neuron. Each section can have multiple 3D points that describe its shape and structure. By summing up the number of 3D points across all sections, we can get an idea of the complexity of the neuron's morphology as represented in the model.

print(cell.id)

#print(cell.size()) 
n = int(cell.id.size()) #By calling cell.id.size(), we can get the total number of points in the SWC file, which is stored in the variable n. This allows us to iterate through all the points and access their corresponding IDs and types for further processing or analysis.

for i in range(n):
    swc_id = int(cell.id.x[i]) # cell.id is a vector that contains the IDs of the points in the SWC file. By accessing cell.id.x[i], we can retrieve the ID of the i-th point. The int() function is used to convert the value to an integer for easier readability and processing.
    swc_type = int(cell.type.x[i]) # cell.type is a vector that contains the types of the points in the SWC file. By accessing cell.type.x[i], we can retrieve the type of the i-th point. The int() function is used to convert the value to an integer for easier readability and processing. The type indicates whether the point corresponds to a soma, axon, dendrite, etc., based on the SWC format specifications.
    print(f"Index {i}: ID={swc_id}, Type={swc_type}") # this will print the index of each point along with its corresponding ID and type. The index is simply the position of the point in the list, while the ID and type provide information about the specific characteristics of that point in the neuron's morphology. This can be useful for understanding the structure of the neuron and for further analysis or visualization.
'''
#---------------------------------------------------------------------------------------------------------------------------------------


print("\nTotal sections:", len(list(h.allsec())))

# code to visualize the morphology
shapes = []  # keep references alive — NEURON GC will close windows if not stored!

for i, cell in enumerate(all_cells):
    sl = h.SectionList()
    for sec in cell.all: # .all is  an attribute of the Cell class that contains all the sections of that cell. By iterating through cell.all, we can access each section and append it to the SectionList (sl) for visualization. This allows us to create a visual representation of the neuron's morphology based on the sections defined in the SWC file.
        sl.append(sec=sec)
    
    shape = h.Shape(sl)
    shape.exec_menu('3D Rotate')
    shape.show(0)
    shape.flush()
    shapes.append(shape)         # critical — prevents garbage collection closing the window

input("\nPress Enter to exit...\n")



#=======================================================================================================================================================================
#                Example 1: from: https://rutgersconnect-my.sharepoint.com/:w:/g/personal/jdd235_scarletmail_rutgers_edu/IQAh7VvfGQa8RbSowlGy6GQrAU_HWpM2j6IpUcMOXtNC3mE
#=======================================================================================================================================================================
'''vol = CloudVolume('precomputed://gs://neuroglancer/zfish_v1/image', use_https=True)
img = vol[51267:51367, 24985:25085, 17072:17172]

# Check what you got
print(f"Downloaded image shape: {img.shape}")
print(f"Data type: {img.dtype}")
print(f"Value range: {img.min()} to {img.max()}")

# If you want to visualize a slice
import matplotlib.pyplot as plt
plt.imshow(img[:, :, 50], cmap='gray')  # Middle Z slice
plt.title('Zebrafish brain slice')
plt.show()'''

#=======================================================================================================================================================================
#                Example 2: from: https://rutgersconnect-my.sharepoint.com/:w:/g/personal/jdd235_scarletmail_rutgers_edu/IQAh7VvfGQa8RbSowlGy6GQrAU_HWpM2j6IpUcMOXtNC3mE
#=======================================================================================================================================================================
'''vol = CloudVolume('gs://neuroglancer/zfish_v1/consensus-20190415', parallel=True, progress=True)
label = 1
mesh = vol.mesh.get(label)

vol.mesh.save(12345) # save 12345 as ./12345.ply on disk
vol.mesh.save([12345, 12346, 12347]) # merge three segments into one file
vol.mesh.save(12345, file_format='obj') # 'ply' and 'obj' are both supported
vol.mesh.get(12345) # return the mesh as vertices and faces instead of writing to disk
vol.mesh.get([ 12345, 12346 ]) # return these two segids fused into a single mesh
vol.mesh.get([ 12345, 12346 ], fuse=False) # return { 12345: mesh, 12346: mesh }
vol.mesh.put(meshes) # works for unsharded legacy only
vol.mesh.delete(segids) # works for unsharded meshes only

mesh.viewer() # Opens GUI. Requires vtk.'''

#=======================================================================================================================================================================
#                Working with segmentation data directly (since meshes aren't available)
#=======================================================================================================================================================================
    
'''print("Loading segmentation data...")
vol = CloudVolume('precomputed://gs://neuroglancer/zfish_v1/consensus-20190415', 
                    use_https=True, parallel=False, progress=True)
print("✓ Volume loaded successfully")

print(f"\nVolume bounds: {vol.bounds}")

# Download a chunk of segmentation data
print("\nFetching segmentation data...")
seg = vol[45000:45200, 28000:28200, 17200:17400]
seg = seg.squeeze()

print(f"Segmentation shape: {seg.shape}")
print(f"Data type: {seg.dtype}")

unique_labels = np.unique(seg)
print(f"Found {len(unique_labels)} unique labels")
print(f"Labels: {unique_labels[:20]}")

# Visualize the segmentation (different from raw image data)
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Show middle slices
axes[0, 0].imshow(seg[:, :, 100], cmap='nipy_spectral')
axes[0, 0].set_title('XY slice - Segmentation (colored by label)')
axes[0, 0].axis('off')

axes[0, 1].imshow(seg[:, 100, :], cmap='nipy_spectral')
axes[0, 1].set_title('XZ slice')
axes[0, 1].axis('off')

axes[1, 0].imshow(seg[100, :, :], cmap='nipy_spectral')
axes[1, 0].set_title('YZ slice')
axes[1, 0].axis('off')

# Show binary mask of one segment
if len(unique_labels) > 1:
    label = unique_labels[1]  # First non-background label
    mask = (seg == label)
    axes[1, 1].imshow(mask[:, :, 100], cmap='gray')
    axes[1, 1].set_title(f'Single neuron (label {label})')
    axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('zebrafish_segmentation.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to 'zebrafish_segmentation.png'")

print("\n" + "="*80)
print("NOTE: This dataset appears to not have pre-computed meshes available.")
print("You've successfully downloaded and visualized the SEGMENTATION data instead.")
print("Each color in the image represents a different neuron/cell.")
print("="*80)

if __name__ == '__main__':
    main()'''

#=======================================================================================================================================================================
#                Visualizing SWC files Dr. Aksay Sent
#=======================================================================================================================================================================
'''def load_swc(filepath):
    """
    Load an SWC file and return the data.
    
    SWC format columns:
    0: ID (sample number)
    1: Type (1=soma, 2=axon, 3=dendrite, 4=apical dendrite, etc.)
    2: X coordinate
    3: Y coordinate
    4: Z coordinate
    5: Radius
    6: Parent ID (-1 for root/soma)
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            # Parse the data
            parts = line.split()
            if len(parts) == 7:
                data.append([float(x) for x in parts])
    
    return np.array(data)

swc_file1 = "/home/aksay_lab/NeuronProject/CloudvolNeuron_Test/76182_reRoot_reSample_5000.swc"
skeleton_id = os.path.basename(swc_file1).split('_')[0] # Extract skeleton ID from filename

def visualize_swc(swc_data, skeleton_id, output_file='neuron_visualization.png'):
    """
    Visualize a neuron from SWC data.
    """
    fig = plt.figure(figsize=(8, 15))
    fig.canvas.manager.set_window_title('Figure 1: skeleton ID:' + skeleton_id)

    # Extract coordinates
    ids = swc_data[:, 0].astype(int)
    types = swc_data[:, 1].astype(int)
    x = swc_data[:, 2] # the : operator is used to select all rows of the specified column 
    # in this case, column index 2 which corresponds to the X coordinate. (indeces in python start at 0)
    y = swc_data[:, 3]
    z = swc_data[:, 4]
    radius = swc_data[:, 5]
    parent_ids = swc_data[:, 6].astype(int) # astype() is a method of numpy arrays, not lists. If parent_ids is a numpy array,
    # you can use astype() to convert its data type. However, if parent_ids is a list, you would need to convert it to a numpy
    # array first before using astype(). Here's how you can do it:
    
    # Color map for different neurite types
    type_colors = {
        1: 'red',      # Soma
        2: 'blue',     # Axon
        3: 'green',    # Basal dendrite
        4: 'purple',   # Apical dendrite
    }
    
    # 3D plot
    ax1 = fig.add_subplot(311, projection='3d')
    
    # Draw connections
    for i, (node_id, parent_id, node_type) in enumerate(zip(ids, parent_ids, types)):
        if parent_id != -1:  # Not root
            # Find parent index
            print("\n Processing node ID:", node_id, "with parent ID:", parent_id)
            parent_idx = np.where(ids == parent_id)[0]
            print('\n Parent index found at:', parent_idx)
            if len(parent_idx) > 0:
                parent_idx = parent_idx[0]
                color = type_colors.get(node_type, 'gray')
                ax1.plot([x[i], x[parent_idx]], 
                        [y[i], y[parent_idx]], 
                        [z[i], z[parent_idx]], 
                        color=color, linewidth=1)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Neuron Structure of ' + os.path.basename(swc_file1).split('_')[0])
    
    # XY projection
    ax2 = fig.add_subplot(312)
    for i, (node_id, parent_id, node_type) in enumerate(zip(ids, parent_ids, types)):
        if parent_id != -1:
            parent_idx = np.where(ids == parent_id)[0]
            if len(parent_idx) > 0:
                parent_idx = parent_idx[0]
                color = type_colors.get(node_type, 'gray')
                ax2.plot([x[i], x[parent_idx]], [y[i], y[parent_idx]], 
                        color=color, linewidth=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    
    # XZ projection
    ax3 = fig.add_subplot(313)
    for i, (node_id, parent_id, node_type) in enumerate(zip(ids, parent_ids, types)):
        if parent_id != -1:
            parent_idx = np.where(ids == parent_id)[0]
            if len(parent_idx) > 0:
                parent_idx = parent_idx[0]
                color = type_colors.get(node_type, 'gray')
                ax3.plot([x[i], x[parent_idx]], [z[i], z[parent_idx]], 
                        color=color, linewidth=1)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to '{output_file}'")
    plt.show()  # ADD THIS LINE to display the window
    
    # Print statistics
    print(f"\nNeuron Statistics:")
    print(f"  Total nodes: {len(swc_data)}")
    print(f"  Node types: {np.unique(types)}")
    for node_type in np.unique(types):
        count = np.sum(types == node_type)
        type_name = {1: 'Soma', 2: 'Axon', 3: 'Dendrite', 4: 'Apical dendrite'}.get(node_type, 'Unknown')
        print(f"    Type {node_type} ({type_name}): {count} nodes")

def main():
    # REPLACE THIS with the path to your SWC file
    #swc_file1 = "/home/aksay_lab/NeuronProject/CloudvolNeuron_Test/76182_reRoot_reSample_5000.swc"
    print(f'Loading SWC file: {swc_file1}')
    #swc_file2 = "~/NeuronProject/CloudvolNeuron_Test/76199_reRoot_reSample_5000.swc"
    #swc_file3 = "~/NeuronProject/CloudvolNeuron_Test/76200_reRoot_reSample_5000.swc"

    print(os.path.basename(swc_file1).split('_')[0]) # this will print the base name of the file without 
    #the directory path, and then split it by the underscore character and take the first part 
    # (which is the ID of the neuron). The '[0]' at the end is used to select the first element 
    # of the resulting list from the split operation.
    swc_data = load_swc(swc_file1)
    
    print("Visualizing neuron...")
    visualize_swc(swc_data, skeleton_id, output_file='76182.png')

if __name__ == '__main__':
    main()'''

