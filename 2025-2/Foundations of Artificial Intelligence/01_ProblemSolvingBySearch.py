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


    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, \
            problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node


