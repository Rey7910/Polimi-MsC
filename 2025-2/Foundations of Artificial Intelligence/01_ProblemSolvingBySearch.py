class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0

        if parent:
            self.depth= parent.depth + 1


