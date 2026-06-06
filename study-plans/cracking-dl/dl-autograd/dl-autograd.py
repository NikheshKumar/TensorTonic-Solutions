import numpy as np

def autograd(operations, input_values):
    """
    Returns: Dict with "output" (float) and "gradients" (list of floats), rounded to 4 decimals.
    """
    class Node:
        def __init__(self, data, prev=()):
            self.data = float(data)
            self.grad = 0.0
            self.prev = prev
            self._backward = lambda: None

    nodes = [Node(v) for v in input_values]
    n_inputs = len(input_values)

    for op in operations:
        if op[0] == "add":
            a, b = nodes[op[1]], nodes[op[2]]
            out = Node(a.data + b.data, (a, b))
            # TODO: define out._backward to propagate gradients
            
            out._backward = lambda a=a, b=b, out=out: (
                setattr(a, 'grad', a.grad + out.grad), 
                setattr(b, 'grad', b.grad + out.grad)
            )
            
            nodes.append(out)
            
        elif op[0] == "mul":
            a, b = nodes[op[1]], nodes[op[2]]
            out = Node(a.data * b.data, (a, b))
            # TODO: define out._backward to propagate gradients
            
            out._backward = lambda a=a, b=b, out=out: (
                setattr(a, 'grad', a.grad + out.grad * b.data), 
                setattr(b, 'grad', b.grad + out.grad * a.data)
            )
            
            nodes.append(out)
            
        elif op[0] == "neg":
            a = nodes[op[1]]
            out = Node(-a.data, (a,))
            # TODO: define out._backward to propagate gradients
            
            def backward_function(a, b, out):
                a.grad -= out.grad
                
            out._backward = lambda a=a, out=out: (
                setattr(a, 'grad', a.grad - out.grad)
            )
            
            nodes.append(out)

    output_node = nodes[-1]
    output_node.grad = 1.0


    # TODO: topological sort and call _backward in reverse order
    seen_nodes= set()
    sorted_nodes = []

    def topological_sorting(n):
        if id(n) not in seen_nodes:
            seen_nodes.add(id(n))
            for p in n.prev:
                topological_sorting(p)
            sorted_nodes.append(n)
    
    topological_sorting(output_node)
    
    for n in reversed(sorted_nodes):
        n._backward()

    gradients = [round(nodes[i].grad, 4) for i in range(n_inputs)]
    return {"output": round(output_node.data, 4), "gradients": gradients}