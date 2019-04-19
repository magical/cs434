import math

# choose best test for root of tree
#       pick test on a feature
# recursively choose tests for the descendent nodes
# if all data belongs to the same class, make a leaf node
#       or just stop when the error is small enough

# test ideas:
#  - pick feature that divides roughlyl in half
#  - divide data so that one side is more positive and one side is more negative


def divide(S):
    for f in features:
        U = []
        for value in f:
            si = split(f, value, S)
            pi = len(si) / len(S)

        benefit = entropy(S) - sum(entropy(si) * pi

    errors = x
    if errors < threshold:
        return None

    best_feature = whatever

    tree = Node(best_feature)
    for value in best_feature:
        si = split(best_feature, value, S)
        subtree = divide(s)
        node.add(value, subtree)

    return


def entropy(S):
    p = {}
    c = Counter(S):
    for v in c:
        p[v] = len(c) / len(S)
    return -sum(p[v]*log2(p[v]) for v in c)

def log2(x):
    return math.log(x, 2)

def get_boundary_points(S, f):
    """find the set of boundary points for a continuous feature"""
    values = f
    values.sort()
    boundary = []
    for v1, v2 in zip(values, values[1:]):
        m = v1 + (v2-v1)/2
        boundary.append(m)
    return boundary


# pruning:
#   build a perfect tree on the training set
#   measure error against validation set
#   divide validtion data based on the tree - pump it to the leaf nodes
#   for each feature node directly above a leaf, compute validtion error change if that node removed
#   remove node that improves validtion error the most
