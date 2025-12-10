import neuron
from neuron import h
'''import neuron.h
import neuron.rdx
import neuron.gui2'''
import pprint
import textwrap
'''from bokeh.io import output_notebook 
import bokeh.plotting as plt''' # for jupyter 
import matplotlib.pyplot as plt
import csv
import plotnine as p9
import pandas as pd
import json
import pickle
import numpy as np
import math
from math import pi

print(neuron.__version__)
from neuron import n 
from neuron.units import ms, mV, um

'''A Section is the basic morphological building-block in NEURON. We typically think of a Section
 as an unbranched cable, but it can also be used to represent a soma. Thus a simple model neuron 
 with only a soma can be created as in:'''

# creat a cell with a single section (soma)
soma = n.Section('soma')

# Aside 1: NEURON’s n.topology function
'''NEURON’s n.topology() function displays the topological structure of the entire model, indicating 
which sections are connected to which sections, where they are connected, and how many segments each 
section is divided into.'''

n.topology() # output = |-|       soma(0-1)

#Aside 2: The psection method (properties section)

'''soma.psection()'''

'''Since this is a dictionary, we can extract any properties we want using square brackets. For example,
 the length of the section is:'''

## soma.psection()["morphology"]["L"]

pprint.pprint(soma.psection())


'''soma.psection()["morphology"]["L"]
print()
pprint.pprint(soma.psection()["morphology"]["L"])
'''
print()
soma.L = 20 # set the length of the section to 20 microns
soma.diam = 20 # set the diameter of the section to 20 microns
print(soma.psection()["morphology"]["L"])
print(soma.psection()["morphology"]["diam"])

print()
dir(soma)
'''
# pprint.pprint(dir(soma))

# print(textwrap.fill(", ".join(dir(n))))

help(soma.connect) '''

#soma.insert(n.hh)
#pprint.pprint(soma.insert(n.hh))

''' # The number of segments within a section is given by the variable, nseg. 
To summarize, we access sections by their name and segments by some location on the section.
Section: section
Segment: section(loc)

#Using the Python type function can tell us what a variable is:
print(f"type(soma) = {type(soma)}")
print(f"type(soma(0.5)) = {type(soma(0.5))}")

# Segment variables follow the idiom:

'section(loc).var' # or 
'section(loc).mech.var' # or 
'section(loc).var_mech' # the first form is preferred 
'''

'''print()
mech = soma(0.5).hh
print(dir(mech))

print()
print(mech.gkbar)
print(soma(0.5).hh.gkbar)'''


iclamp = n.IClamp(soma(0.5)) # create an IClamp object at the center of the soma
'''An IClamp is a Point Process. Point processes are point sources of current. 
When making a new PointProcess, you pass the segment to which it will bind.'''

print()
#print([item for item in dir(iclamp) if not item.startswith("__")])
'''In particular, we notice three key properties of a current clamp: amp – the amplitude (in nA),
delay – the time the current clamp switches on (in ms), and dur – how long (in ms) the current clamp stays on.
Let’s set these values:'''

iclamp.delay = 0
iclamp.dur = 15
iclamp.amp = 0.9


soma.psection()
print(soma.psection())           
# pprint.pprint(soma.psection())    

''' we will record the membrane potential, which is soma(0.5).v and the corresponding 
time points (n.t). References to variables are available by preceding the last part of the 
variable name with a _ref_'''
print()
v = n.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
t = n.Vector().record(n._ref_t)  # Time stamp vector

'''NEURON h module provides the low level fadvance function for advancing one time step. 
For higher-level simulation control specification, we load NEURON’s stdrun library'''
n.load_file("stdrun.hoc")
n.finitialize(-65 * mV) # initialize the resting membrane potential to -65 mV

n.continuerun(40 * ms) #continue the simulation from the current time (0) until 40 ms:

'''f = plt.figure(x_axis_label="t (ms)", y_axis_label="v (mV)")
f.line(t, v, line_width=2)
plt.show(f)''' # for jupypter 

#plt.figure()
#plt.plot(t, v)
#plt.xlabel("t (ms)")
#plt.ylabel("v (mV)")
#plt.show()

# Step 9: Saving and loading results
'''The csv (comma separated variables) file format is widely used for data interchange, 
and can be used to transfer data to MATLAB, Excel, etc without writing any special conversion code.'''

# writting the data to a csv file
#with open("data.csv", "w") as f:
#    csv.writer(f).writerows(zip(t, v))

# reading the data from a csv file
#with open("data.csv") as f:
 #   reader = csv.reader(f)
#  tnew, vnew = zip(*[[float(val) for val in row] for row in reader if row])
'''The argument to the zip is a nested list comprehension; the zip and the asterisk together effectively 
transpose the data turning it from a list of (t, v) pairs into a list of t values and a list of v values. 
For loading more variables, the right hand side of the last line is unchanged; all that changes is that
 the variables need to be listed on the left; e.g. tnew, vnew, canew = zip(…)
We can plot our newly loaded data (here with matplotlib) to see that it is the same as before:'''

#plt.figure()
#plt.plot(tnew, vnew)
#plt.xlabel("t (ms)")
#plt.ylabel("v (mV)")
#plt.show()

'''If the CSV file had a header row identifying the columns, then pd.read_csv would have handled the 
names automatically and we would not have had to specify the last two arguments above.)
And now plot with plotnine’s version of ggplot. This function provides an implementation of Wilkinson’s 
Grammar of Graphics; as such, the interface is essentially identical to the R function of the same 
name.'''

#data = pd.read_csv("data.csv", header=None, names=["t", "v"])
#g = (p9.ggplot(data, p9.aes(x="t", y="v")) + p9.geom_path())
#g.save("plot.png")

## Alternative: Using JSON format
''' Here we built a dictionary with keys t and v, and stored their values as a list. Since JSON 
is a language-independent format, it does not have a concept of NEURON Vectors, which is why we had 
to create a list copy of them before saving. The indent=4 argument is optional, but indents the output 
to make it more human-readable (at the cost of a larger file size).# Writting data to a JSON file

with open("data.json", "w") as f:
    json.dump({"t": list(t), "v": list(v)}, f, indent=4)

# Reading data from a JSON file
with open("data.json") as f:
    data = json.load(f)
tnew = data["t"]
vnew = data["v"]

plt.figure()
plt.plot(tnew, vnew)
plt.xlabel("t (ms)")
plt.ylabel("v (mV)")
plt.show()'''

## Alternative: Using Python pickle format
'This is slightly cleaner than the JSON solution above because it is Python specific and '
'therefore able to explicitly encode NEURON Vector objects.'
# writting
'''with open("data.p", "wb") as f:
    pickle.dump({"t": t, "v": v}, f)

# reading
with open("data.p", "rb") as f:
    data = pickle.load(f)
tnewp = data["t"]
vnewp = data["v"]
'Pickles in Python 3 are by default binary files, so we have to specify write and read flags of '
'wb and rb respectively. We use a different variable name here than before simply to indicate that '
'what is loaded in has type'

type(tnewp) # (hoc.Vector)

tnewp.hname() # and in particular is a NEURON Vector (Vector[2])

type(tnew) # Unlike the other solutions provided, which construct regular Python lists (list)

'This is a minor distinction though, as we’ve already seen list(vec) copies a Vector vec into a new list.'
'Using n.Vector(old_list) makes a NEURON Vector that is a copy of old_list.'

plt.figure()
plt.plot(tnewp, vnewp)
plt.xlabel("t (ms)")
plt.ylabel("v (mV)")
plt.show()'''

