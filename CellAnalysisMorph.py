from neuron import h
import itertools
import pprint

class Cell:
    def __init__(self):
        self.load_morphology()
    def load_morphology(self):
        h.load_file('import3d.hoc')
        cell = h.Import3d_SWC_read()
        cell.input('76182.swc')
        i3d = h.Import3d_GUI(cell, 0)
        i3d.instantiate(self)

'''all_diams = list(itertools.chain.from_iterable(
    [[sec.diam3d(i) for i in range(sec.n3d())] for sec in h.allsec()]
))

print(f'fMinimum diameter: {min(all_diams)}')  
print(f'Maximum diameter: {max(all_diams)}')'''


if __name__ == "__main__":
    cell = Cell()
    h.topology()

