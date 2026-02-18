#=======================================================================================================================================================================
#          Manipulating SWC files and meshes with CloudVolume
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
import os # os stands for operating system, and it's a built-in Python module that provides a way to interact with 
# the underlying operating system. It allows you to perform various tasks such as file and directory manipulation, 
# environment variable access, and more. In this code, we will use the os module to check if the specified SWC file 
# exists before attempting to load it.

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

swc_path = 'SWC_files/your_file.swc'  # adjust!
print("File exists?", os.path.exists(swc_path))

cell = h.Import3d_SWC_read()
cell.quiet = 0  # show warnings
cell.input(swc_path)

print("Parsed sections:", cell.sections.count())
print("Total points:", cell.total_points)

# If count > 0, proceed
i3d = h.Import3d_GUI(cell, 0)
i3d.instantiate(None)

print("After instantiate - sections in model:", len(list(h.allsec())))
for sec in h.allsec():
    print(h.secname(sec=sec))
h.topology()
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

