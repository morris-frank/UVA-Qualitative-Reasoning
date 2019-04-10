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
    exogenous: bool


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
    Quantity('Inflow', UnboundedMagnitude,  Derivative, True),
    Quantity('Volume', BoundedMagnitude,  Derivative, False),
    Quantity('Outflow', BoundedMagnitude,  Derivative, False)
]

quantities = [
    Quantity('Inflow', UnboundedMagnitude,  Derivative, True),
    Quantity('Volume', BoundedMagnitude,  Derivative, False),
    Quantity('Outflow', BoundedMagnitude,  Derivative, False),
    Quantity('Height', BoundedMagnitude,  Derivative, False),
    Quantity('Pressure', BoundedMagnitude,  Derivative, False),
]

dependencies = [
    VCDependency('Volume', 'Outflow', 'MAX', 'MAX'),
    VCDependency('Volume', 'Outflow', 'ZERO', 'ZERO'),
    PDependency('Volume', 'Outflow'),
    IDependency('Inflow', 'Volume', 'POS'),
    IDependency('Outflow', 'Volume', 'NEG')
]

dependencies = [
    VCDependency('Volume', 'Outflow', 'MAX', 'MAX'),
    VCDependency('Volume', 'Outflow', 'ZERO', 'ZERO'),
    VCDependency('Pressure', 'Outflow', 'MAX', 'MAX'),
    VCDependency('Pressure', 'Outflow', 'ZERO', 'ZERO'),
    IDependency('Inflow', 'Volume', 'POS'),
    IDependency('Outflow', 'Volume', 'NEG'),
    PDependency('Pressure', 'Outflow'),
    PDependency('Volume', 'Height'),
    PDependency('Height', 'Pressure')
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

def edge_filter(dependencies):
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
        for idependency in filter(lambda x:x.__class__==IDependency, dependencies):
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

        for  i in range(len(old)):
            oldMag, oldDrv = old[i]
            newMag, newDrv = new[i]
            if quantities[i].exogenous or oldDrv == newDrv:
                continue
            if oldDrv == 'POS' and newDrv == 'ZERO' and newMag == 'MAX':
                continue
            if oldDrv == 'NEG' and newDrv == 'ZERO' and newMag == 'ZERO':
                continue
            if i in influenceSums:
                if influenceSums[i] is not None or influenceSums[i] == newDrv:
                    edge_labels[pair] = True
                    continue
            cont = False
            for dependency in filter(lambda x:x.__class__==PDependency, dependencies):
                if quantities[i].name == dependency.left_name and new[qIdx[dependency.right_name]][1] == newDrv:
                    cont = True
                if quantities[i].name == dependency.right_name and new[qIdx[dependency.left_name]][1] == newDrv:
                    cont = True
            if cont:
                continue
            return False

        for dependency in filter(lambda x:x.__class__==PDependency, dependencies):
            i_left = qIdx[dependency.left_name]
            i_right = qIdx[dependency.right_name]
            if old[i_left][1] == new[i_left][1] and old[i_left][1] == new[i_right][1]:
                continue
            if quantities[i_left].exogenous or quantities[i_right].exogenous:
                continue
            if i_left in influenceSums or i_right in influenceSums:
                continue
            return False
        return True
    return _filter


def repr_state(s):
    rs = {'MAX': 'max', 'POS': '+', 'NEG': '-', 'ZERO': '0'}
    return 'In[{},{}]\nVol[{},{}]\nOut[{},{}]'.format(rs[s[0][0]], rs[s[0][1]], rs[s[1][0]], rs[s[1][1]], rs[s[2][0]], rs[s[2][1]])


qIdx = {q.name:i for i,q in enumerate(quantities)}

quantity_states = [product(q.magnitude._member_names_,q.derivative._member_names_) for q in quantities]
filtered_quantity_states = [filter(quantity_filter, qs) for qs in quantity_states]

node_states = list(product(*filtered_quantity_states))
filtered_node_states = list(filter(node_filter(dependencies), node_states))


edge_labels = {}
edge_states = list(product(filtered_node_states, filtered_node_states))
filtered_edge_states = list(filter(edge_filter(dependencies), edge_states))


def save_graph(nodes, edges):
    G = pgv.AGraph(
        directed=True,
        overlap=False,
        splines=True,
        sep=+1.2,
        normalize=True,
        smoothing='avg_dist'
    )
    G.add_edges_from(edges)
    for edge,label in edge_labels.items():
        if G.has_edge(edge[0], edge[1]):
            G.get_edge(edge[0], edge[1]).attr['label'] = label
    for edge in edges:
        old,new = edge
        labels = []
        for i in range(len(old)):
            if new[i][0] != old[i][0] and old[i][1] != 'ZERO':
                labels.append('∂')
            if quantities[i].exogenous and old[i][1] != new[i][1]:
                labels.append('exo')
        if edge in edge_labels:
            labels.append('I')
        G.get_edge(edge[0], edge[1]).attr['label'] = ','.join(labels)
    for node in nodes:
        if G.has_node(node):
            G.get_node(node).attr['label'] = repr_state(node)
            for dependency in filter(lambda x:x.__class__==VCDependency, dependencies):
                    if node[qIdx[dependency.left_name]][0] == dependency.left_mag:
                        G.get_node(node).attr['style'] = 'dashed'
                        G.get_node(node).attr['color'] = 'blue'
    G.draw('test.pdf', prog='dot')

save_graph(filtered_node_states, filtered_edge_states)

def main():
    print('Starting simulating')

if __name__ == "__main__":
    main()
