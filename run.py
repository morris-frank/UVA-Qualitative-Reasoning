#!/bin/python
from graphbuilder import Quantity, VCDependency, IDependency, PDependency, Derivative, UnboundedMagnitude, BoundedMagnitude
from graphbuilder import simulate, user_trace


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
        VCDependency('Volume', 'Height', 'MAX', 'MAX'),
        VCDependency('Volume', 'Height', 'ZERO', 'ZERO'),
        VCDependency('Height', 'Pressure', 'MAX', 'MAX'),
        VCDependency('Height', 'Pressure', 'ZERO', 'ZERO'),
        VCDependency('Pressure', 'Outflow', 'MAX', 'MAX'),
        VCDependency('Pressure', 'Outflow', 'ZERO', 'ZERO'),
        IDependency('Inflow', 'Volume', 'POS'),
        IDependency('Outflow', 'Volume', 'NEG'),
        PDependency('Volume', 'Height'),
        PDependency('Height', 'Pressure'),
        PDependency('Pressure', 'Outflow')
    ]
}


def main():
    print('RUNNING without extras')
    graph = simulate(NO_EXTRAS['quantities'], NO_EXTRAS['dependencies'])
    graph.draw('no_extras.pdf', prog='dot')
    user_trace(graph, NO_EXTRAS['quantities'])

    graph = simulate(WITH_EXTRAS['quantities'], WITH_EXTRAS['dependencies'])
    graph.draw('with_extras.pdf', prog='dot')


if __name__ == "__main__":
    main()
