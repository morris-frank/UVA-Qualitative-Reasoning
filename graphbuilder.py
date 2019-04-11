from dataclasses import dataclass
from enum import Enum
from typing import Union, Callable, List, Tuple, Iterable, Dict
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


Dependency = Union[VCDependency, PDependency, IDependency]
Quantity_State = Tuple[str, str]
Node_State = Iterable[Quantity_State]
Edge_State = Tuple[Node_State, Node_State]


def quantity_filter(quantity: Quantity_State) -> bool:
    """Filters out invalid single states of a quantity.

    Arguments:
        quantity {Quantity_State} -- The quantity datapoint

    Returns:
        bool -- whether valid
    """
    mag,drv = quantity
    if (mag == 'MAX'  and drv == 'POS') or \
       (mag == 'ZERO' and drv == 'NEG'):
        return False
    return True


def node_filter(quantities: List[Quantity], dependencies: List[Dependency]) -> Callable[[Node_State], bool]:

    """Returns a filter that filters out node according to the dependencies given.

    Arguments:
        quantities {List[Quantity]} -- The list of quantities
        dependencies {List[Dependency]} -- The list of dependencies

    Returns:
        Callable -- the callable filter
    """
    quantity_indexes = {q.name:i for i,q in enumerate(quantities)}
    def _filter(node: Node_State) -> bool:
        """the filter

        Arguments:
            node {Iterable[Quantity_State]} -- the list of all states of quantities in the node

        Returns:
            bool -- whether its valid
        """
        for dependency in filter(lambda x:x.__class__ in [VCDependency, PDependency], dependencies):
            left_idx = quantity_indexes[dependency.left_name]
            right_idx = quantity_indexes[dependency.right_name]
            if dependency.__class__ == VCDependency:
                left_mag, right_mag = node[left_idx][0], node[right_idx][0]
                if  (left_mag == dependency.left_mag) ^ (right_mag == dependency.right_mag):
                    return False
            elif dependency.__class__ == PDependency:
                left_drv, right_drv = node[left_idx][1], node[right_idx][1]
                if left_drv != right_drv:
                    return False
        return True
    return _filter


def edge_filter(quantities: List[Quantity], dependencies: List[Dependency], influenced_edges: Dict[Edge_State, bool]) -> Callable[[Edge_State], bool]:
    """Returns a filter that filters out edges (pair of node states) that are in conflict with the given dependencies

    Arguments:
        quantities {List[Quantity]} -- the list of quantities
        dependencies {List[Dependency]} -- the list of dependencies to adhere to

    Returns:
        Callable[[Edge_State], bool] -- the filter
    """
    quantity_indexes = {q.name:i for i,q in enumerate(quantities)}
    def _filter(pair: Edge_State) -> bool:
        """the filter

        Arguments:
            pair {Edge_State} -- the pair of node states

        Returns:
            bool -- whether the edge is valid
        """
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
        directed_influences = {}
        for idependency in filter(lambda x:x.__class__==IDependency, dependencies):
            startIdx, endIdx = quantity_indexes[idependency.start], quantity_indexes[idependency.end]
            if old[startIdx][0] == 'ZERO':
                continue
            directed_influences.setdefault(endIdx, idependency.sign)
            if directed_influences[endIdx] != idependency.sign:
                directed_influences[endIdx] = None
        for endIdx, sign in directed_influences.items():
            if sign is None:
                continue
            if (old[endIdx][1] == sign) or \
               (old[endIdx][1] == 'ZERO'):
                if new[endIdx][1] != sign:
                    return False

        for  i in range(len(old)):
            oldMag, oldDrv = old[i]
            newMag, newDrv = new[i]
            if (quantities[i].exogenous or oldDrv == newDrv) or \
               (oldDrv == 'POS' and newDrv == 'ZERO' and newMag == 'MAX') or \
               (oldDrv == 'NEG' and newDrv == 'ZERO' and newMag == 'ZERO'):
                continue
            if i in directed_influences:
                if directed_influences[i] is not None or directed_influences[i] == newDrv:
                    influenced_edges[pair] = True
                    continue
            cont = False
            for dependency in filter(lambda x:x.__class__==PDependency, dependencies):
                leftIdx,rightIdx = quantity_indexes[dependency.left_name], quantity_indexes[dependency.right_name]
                if (quantities[i].name == dependency.left_name and new[leftIdx][1] == newDrv) or \
                   (quantities[i].name == dependency.right_name and new[rightIdx][1] == newDrv):
                    cont = True
            if cont:
                continue
            return False

        for dependency in filter(lambda x:x.__class__==PDependency, dependencies):
            leftIdx, rightIdx = quantity_indexes[dependency.left_name], quantity_indexes[dependency.right_name]
            if (old[leftIdx][1] == new[leftIdx][1] and old[leftIdx][1] == new[rightIdx][1]) or \
               (quantities[leftIdx].exogenous or quantities[rightIdx].exogenous) or \
               (leftIdx in directed_influences or rightIdx in directed_influences):
                continue
            return False
        return True
    return _filter


def repr_func(quantities: List[Quantity]) -> Callable[[Node_State], str]:
    """Gives a function that can stringify a node given our quantaties

    Arguments:
        quantities {List[Quantity]} -- the quantitites

    Returns:
        Callable[[Node_State], str] -- the repr function
    """
    sq = [q.name[:3] for q in quantities]
    sv = {'MAX': 'max', 'POS': '+', 'NEG': '-', 'ZERO': '0'}
    def _repr(node: Node_State):
        rlist = ['{}[{},{}]'.format(sq[i], sv[mag], sv[drv]) for i,(mag,drv) in enumerate(node)]
        return '\n'.join(rlist)
    return _repr


def build_graph(nodes: List[Node_State], edges: List[Edge_State], quantities: List[Quantity], dependencies: List[Dependency], influenced_edges: Dict[Edge_State, bool]) -> pgv.AGraph:
    """Build the graph from the list of nodes and edges using pygraph

    Arguments:
        nodes {List[Nodes]} -- the list of node states
        edges {List[Edges]} -- the list of directed edges
        quantities {List[Quanitty]} -- tje list of quantities
        dependencies {List[Dependency]} -- the dependencies
        influenced_edges {Dict[Edge_State, bool]} -- contains all the edges that where influenced by a I dependency

    Returns:
        pgv.AGraph -- the pygraph Graph
    """
    quantity_indexes = {q.name:i for i,q in enumerate(quantities)}
    G = pgv.AGraph(
        directed=True,
        overlap=False,
        splines=True,
        sep=+1.2,
        normalize=True,
        smoothing='avg_dist',
        nodesep=0.2,
        ranksep=0.01,
        pack=True
    )

    G.add_edges_from(edges, fontsize=20)

    # Build the  edge labels
    for edge in edges:
        old,new = edge
        labels = set()
        for i in range(len(old)):
            if new[i][0] != old[i][0] and old[i][1] != 'ZERO':
                labels.add('∂')
            if quantities[i].exogenous and old[i][1] != new[i][1]:
                labels.add('↘')
        if edge in influenced_edges:
            labels.add('I')
        G.get_edge(old, new).attr['label'] = ','.join(sorted(labels))

    # Adjust node styles
    _repr = repr_func(quantities)
    for node in nodes:
        if not G.has_node(node):
            continue
        _node = G.get_node(node)
        _node.attr['style'] = 'filled,solid'
        _node.attr['fillcolor'] = 'moccasin'
        _node.attr['label'] = _repr(node)
        for dependency in filter(lambda x:x.__class__==VCDependency, dependencies):
                if node[quantity_indexes[dependency.left_name]][0] == dependency.left_mag:
                    _node.attr['style'] = 'filled,dotted'
                    _node.attr['fillcolor'] = 'khaki'
    return G


def simulate(quantities: List[Quantity], dependencies: List[Dependency]) -> pgv.AGraph:
    quantity_states = [product(q.magnitude._member_names_,q.derivative._member_names_) for q in quantities]
    filtered_quantity_states = [filter(quantity_filter, qs) for qs in quantity_states]

    nodes = list(product(*filtered_quantity_states))
    _node_filter = node_filter(quantities, dependencies)
    filtered_nodes = list(filter(_node_filter, nodes))

    edges = list(product(filtered_nodes, filtered_nodes))
    influenced_edges = {}
    _edge_filter = edge_filter(quantities, dependencies, influenced_edges)
    filtered_edges = list(filter(_edge_filter, edges))
    graph = build_graph(filtered_nodes, filtered_edges, quantities, dependencies, influenced_edges)
    return graph
