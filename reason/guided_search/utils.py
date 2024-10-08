from .tree import Node


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node
