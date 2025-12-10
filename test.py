
class test:
    def __init__(self):
        self.v = 0

test1 = test()
for test1.v in range(5):
    print(test1.v)

#-------------------------------------------------------------------------------------------
#                                              4 Different Pulse Magnitudes     
#-------------------------------------------------------------------------------------------

Rn = 20   # Membrane resistance
diam = [0.01 * k for k in range(1, 5)]  # Diameter list
colors = ["green", "blue", "red", "black"]
Rm = [Rn * 4 * np.pi * (d/2)**2 for d in diam]

# Create figure with 1 row and 2 columns (side by side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# LEFT PLOT: Voltage vs Time (superimposed)
for d, Rm_val, color in zip(diam, Rm, colors):
    soma.diam = d
    amp = 0.075 * Rm_val
    iclamp.amp = amp # to introduce the amperage in the simulation
    
    # Recreate recording vectors for each simulation
    v = n.Vector().record(soma(0.5)._ref_v)
    t = n.Vector().record(n._ref_t)
    
    n.finitialize(-65 * mV)
    n.continuerun(25 * ms)
    
    # Plot voltage on left subplot
    ax1.plot(list(t), list(v), color=color, linewidth=2, 
             label=f"diam={d:.2f} µm, Rm={Rm_val:.2f} Ω-cm², amp={amp:.3f} nA")
ax1.set_xlim(0, 25)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Voltage (mV)")
ax1.set_title("Soma Voltage vs Time")
ax1.legend(fontsize=8)
ax1.grid(True)

# RIGHT PLOT: Unit Step Functions (2x2 grid)
# Create sub-axes within the right subplot
gs = ax2.get_subplotspec().subgridspec(2, 2, hspace=0.3, wspace=0.3) 
step_axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

for idx, (d, Rm_val, color) in enumerate(zip(diam, Rm, colors)): # enrumarete: index + value
    '''zip(diam, Rm, colors): This part combines the lists diam, Rm, and colors into tuples, pairing up their 
    respective elements at the same index.​
    enumerate(...): The enumerate function adds an index counter to each tuple from zip(), producing a sequence 
    like (idx, (d, Rm_val, color)) where idx is the index and d, Rm_val, and color are the corresponding elements 
    from the original lists.'''
    soma.diam = d
    amp = 0.075 * Rm_val
    iclamp.amp = amp
    
    # Recreate recording vectors for time
    t = n.Vector().record(n._ref_t)
    
    n.finitialize(-65 * mV)
    n.continuerun(4 * ms)
    
    # Create unit step function
    t_array = np.array(list(t))
    step_function = np.where((t_array >= iclamp.delay) & (t_array < iclamp.delay + iclamp.dur), 
                             amp, 0) #where : if condition is true, set to amp, else 0
    
    # Plot on corresponding subplot
    step_axes[idx].plot(t_array, step_function, color=color, linewidth=2)
    step_axes[idx].set_title(f"diam={d:.4f} µm\nRm={Rm_val:.6f} Ω-cm²\namp={amp:.6f} nA", 
                             fontsize=8)
    step_axes[idx].grid(True)
    
    # Add labels
    if idx >= 2:
        step_axes[idx].set_xlabel("Time (ms)", fontsize=9)
    if idx % 2 == 0:
        step_axes[idx].set_ylabel("Current (nA)", fontsize=9)

# Remove the right subplot axis (it's just a container)
ax2.axis('off')

fig.suptitle("Voltage Response and Unit Step Functions", fontsize=14, y=0.98)
plt.tight_layout()
plt.show()

print(soma.cm)    

