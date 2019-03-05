from itertools import permutations
dataset = [
    [1, 2, 5],
	[2, 4],
    [2, 3],
    [1, 2, 4],
    [1, 3],
    [2, 3],
    [1, 3],
    [1, 2, 3, 5],
    [1, 2, 3]
]

def algo(data_set, min_support=2):
    data = [row for row in dataset]
    item_set = set([element for row in dataset for element in row])
    candidate_set_i = 1
    candidate_sets = {}
    while 1:
        candidate_set = {}
        _permutations = permutations(item_set, candidate_set_i)
        _permutations = map(lambda x: tuple(sorted(x)), _permutations)  # removing all repeated permutations i.e rearrangement.
        for perm in _permutations:
            count = 0
            for row in data:
                if set(perm).issubset(row):
                    count += 1
            if count >= min_support:
            	candidate_set[perm] = count
        candidate_sets[candidate_set_i] = candidate_set
        candidate_set_i += 1
        if len(candidate_set.keys()) <= 2:
             break
    return candidate_sets

print(algo(dataset))
