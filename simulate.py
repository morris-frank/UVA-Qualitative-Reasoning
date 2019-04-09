#!/bin/python
from dataclasses import dataclass
from enum import Enum
from typing import Union
from itertools import product
import pygraphviz as pgv


class Derivative(Enum):
    NEG = -1
    ZERO = 0
    POS = 1

class BoundedMagnitude(Enum):
    ZERO = 0
    POS = 1
    MAX = 2


class UnboundedMagnitude(Enum):
    ZERO = 0
    POS = 1


DClass = Union[Derivative]
MClass = Union[BoundedMagnitude, UnboundedMagnitude]


@dataclass
class Quantity(object):
    name: str
    magnitude: MClass
    derivative: DClass


@dataclass
class VCDependency(object):
    left_name: str
    right_name: str
    left_mag: str
    right_mag: str


@dataclass
class PDependency(object):
    left_name: str
    right_name: str


@dataclass
class IDependency(object):
    start: str
    end: str
    sign: str


quantities = [
    Quantity('Inflow', UnboundedMagnitude,  Derivative),
    Quantity('Volume', BoundedMagnitude,  Derivative),
    Quantity('Outflow', BoundedMagnitude,  Derivative)
]


dependencies = [
    VCDependency('Volume', 'Outflow', 'MAX', 'MAX'),
    VCDependency('Volume', 'Outflow', 'ZERO', 'ZERO'),
    PDependency('Volume', 'Outflow'),
    IDependency('Inflow', 'Volume', 'POS'),
    IDependency('Outflow', 'Volume', 'NEG')
]


def quantity_filter(quantity):
    mag,drv = quantity
    if (mag == 'MAX'  and drv == 'POS') or \
       (mag == 'ZERO' and drv == 'NEG'):
        return False
    return True


def node_filter(dependencies):
    def _filter(node_state):
        for dependency in dependencies:
            if dependency.__class__ == VCDependency:
                left_mag = node_state[qIdx[dependency.left_name]][0]
                right_mag = node_state[qIdx[dependency.right_name]][0]
                if  (left_mag == dependency.left_mag) ^ (right_mag == dependency.right_mag):
                    return False
            elif dependency.__class__ == PDependency:
                left_drv = node_state[qIdx[dependency.left_name]][1]
                right_drv = node_state[qIdx[dependency.right_name]][1]
                if left_drv != right_drv:
                    return False
        return True
    return _filter


def edge_filter(idependencies):
    def _filter(pair):
        old, new = pair

        # Filter same states
        if old == new:
            return False

        # Filter some value discontinuities
        # · Have positive derivative and end up at zero
        # · Have negative derivative and end up at max
        # · Jump from value zero to max
        # · Jump from value max to zero
        # · Jump from positive derivative to negative
        # · Jump from negative derivative to positive
        # · If we have a derivative the value cannot stay the same
        for i in range(len(old)):
            oldMag, oldDrv = old[i]
            newMag, newDrv = new[i]
            if (oldDrv == 'POS'  and newMag == 'ZERO') or \
               (oldDrv == 'NEG'  and newMag == 'MAX' ) or \
               (oldMag == 'ZERO' and newMag == 'MAX' ) or \
               (oldMag == 'MAX'  and newMag == 'ZERO') or \
               (oldDrv == 'POS'  and newDrv == 'NEG' ) or \
               (oldDrv == 'NEG'  and newDrv == 'POS' ) or \
               (oldDrv == 'ZERO' and oldMag != newMag):
                return False

        # Filter out false influences
        influenceSums = {}
        for idependency in idependencies:
            if old[qIdx[idependency.start]][0] == 'ZERO':
                continue
            endIdx = qIdx[idependency.end]
            influenceSums.setdefault(endIdx, idependency.sign)
            if influenceSums[endIdx] != idependency.sign:
                influenceSums[endIdx] = None
        for endIdx, sign in influenceSums.items():
            if sign is None:
                continue
            if (old[endIdx][1] == sign) or \
               (old[endIdx][1] == 'ZERO'):
                if new[endIdx][1] != sign:
                    return False

        # for i in range(len(old)):
        #     if i in influenceSums:
        #         continue
        #     if old[i][1] != new[i][1]:
        #         if not ((old[i][1] == 'POS' and new[i][1] == 'ZERO' and new[i][0] == 'MAX') or \
        #                 (old[i][1] == 'NEG' and new[i][1] == 'ZERO' and new[i][0] == 'ZERO')):
        #             return False
        return True
    return _filter


def repr_state(s):
    rs = {'MAX': 'max', 'POS': '+', 'NEG': '-', 'ZERO': '0'}
    return 'In[{},{}]\nVol[{},{}]\nOut[{},{}]'.format(rs[s[0][0]], rs[s[0][1]], rs[s[1][0]], rs[s[1][1]], rs[s[2][0]], rs[s[2][1]])


def save_graph(nodes, edges):
    G = pgv.AGraph(
        directed=True,
        overlap=False,
        splines=True,
        sep=+1.2,
        normalize=True,
        smoothing='avg_dist'
    )
    G.add_edges_from(filtered_edge_states)
    for node in nodes:
        if G.has_node(node):
            G.get_node(node).attr['label'] = repr_state(node)
    G.draw('test.pdf', prog='circo')


idependencies =  list(filter(lambda x: x.__class__ == IDependency, dependencies))
qIdx = {q.name:i for i,q in enumerate(quantities)}

quantity_states = [product(q.magnitude._member_names_,q.derivative._member_names_) for q in quantities]
filtered_quantity_states = [filter(quantity_filter, qs) for qs in quantity_states]

node_states = list(product(*filtered_quantity_states))
filtered_node_states = list(filter(node_filter(dependencies), node_states))

edge_states = list(product(filtered_node_states, filtered_node_states))
filtered_edge_states = list(filter(edge_filter(idependencies), edge_states))

save_graph(filtered_node_states, filtered_edge_states)

def main():
    print('Starting simulating')

if __name__ == "__main__":
    main()
