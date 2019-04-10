#!/bin/python
from graphbuilder import Quantity, VCDependency, IDependency, PDependency, Derivative, UnboundedMagnitude, BoundedMagnitude
from graphbuilder import simulate

NO_EXTRAS = {
    'quantities': [
        Quantity('Inflow', UnboundedMagnitude,  Derivative, True),
        Quantity('Volume', BoundedMagnitude,  Derivative, False),
        Quantity('Outflow', BoundedMagnitude,  Derivative, False)
    ],
    'dependencies':[
        VCDependency('Volume', 'Outflow', 'MAX', 'MAX'),
        VCDependency('Volume', 'Outflow', 'ZERO', 'ZERO'),
        PDependency('Volume', 'Outflow'),
        IDependency('Inflow', 'Volume', 'POS'),
        IDependency('Outflow', 'Volume', 'NEG')
    ]
}

WITH_EXTRAS = {
    'quantities': [
        Quantity('Inflow', UnboundedMagnitude,  Derivative, True),
        Quantity('Volume', BoundedMagnitude,  Derivative, False),
        Quantity('Outflow', BoundedMagnitude,  Derivative, False),
        Quantity('Height', BoundedMagnitude,  Derivative, False),
        Quantity('Pressure', BoundedMagnitude,  Derivative, False),
    ],
    'dependencies': [
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
}

def main():
    graph = simulate(NO_EXTRAS['quantities'], NO_EXTRAS['dependencies'])
    graph.draw('no_extras.pdf', prog='dot')

    graph = simulate(WITH_EXTRAS['quantities'], WITH_EXTRAS['dependencies'])
    graph.draw('with_extras.pdf', prog='dot')


if __name__ == "__main__":
    main()
