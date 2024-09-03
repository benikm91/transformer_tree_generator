import abc
import concurrent
import hashlib
import math
import pickle
import random
from array import array
from collections import deque
from dataclasses import dataclass, field
from functools import reduce
from itertools import zip_longest
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class Node:
    def __init__(self, value, children):
        self.value = value
        self.children = children

    def __repr__(self):
        return f"Node({self.value}, {self.children})"

    @staticmethod
    def flatten_tree_bfs(root: 'Node') -> List[int]:
        result = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            result.append(node.value)
            queue.extend(node.children)
        return result

    @staticmethod
    def unflatten_tree_dfs(flattened_tree: list[int], depth: int, branch_factor: int) -> 'Node':
        # Create the root node
        root = Node(flattened_tree[0], [])
        stack = [root]  # Initialize the stack with the root node

        for value in flattened_tree[1:]:
            new_node = Node(value, [])
            stack[-1].children.append(new_node)
            if len(stack) < depth:
                stack.append(new_node)

            while len(stack) > 0 and len(stack[-1].children) == branch_factor:
                stack.pop()  #

        return root

    @staticmethod
    def unflatten_tree_bfs(flattened_tree: list[int], branch_factor: int) -> 'Node':
        root = Node(flattened_tree[0], [])
        queue = deque([root])
        for value in flattened_tree[1:]:
            new_node = Node(value, [])
            if len(queue[0].children) < branch_factor:
                queue[0].children.append(new_node)
                queue.append(new_node)
            else:
                queue.popleft()
                queue[0].children.append(new_node)
                queue.append(new_node)
        return root


    @staticmethod
    def flatten_tree_dfs(root: 'Node') -> List[int]:
        result = []
        stack = [root]
        while stack:
            node = stack.pop()
            result.append(node.value)
            stack.extend(reversed(node.children))  # Reversed to maintain the correct order
        return result


@dataclass
class SampleGen(abc.ABC):

    branch_factor: int = field(init=True)
    depth: int = field(init=True)

    def __post_init__(self):
        assert self.branch_factor > 0 and self.depth > 0, 'All parameters must be positive integers'


    @property
    def number_of_branching_nodes(self) -> int:
        return sum([self.branch_factor ** i for i in range(self.depth)])

    @property
    def number_of_nodes(self) -> int:
        return self.number_of_branching_nodes + self.branch_factor ** self.depth

    @property
    def number_of_leaves(self) -> int:
        return self.branch_factor ** self.depth

    @abc.abstractmethod
    def full_name(self) -> str:
        pass

    def generate_sample(self) -> List[int]:
        res = self._generate_sample()
        assert len(res) == self.number_of_nodes, f"Generated sample has {len(res)} nodes, but {self.number_of_nodes} expected"
        return res

    @abc.abstractmethod
    def _generate_sample(self) -> List[int]:
        pass

    @abc.abstractmethod
    def tree_correct(self, tree_sequence: List[int], error_to_false: bool=True) -> bool:
        pass

    @property
    @abc.abstractmethod
    def best_possible_loss(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def number_of_input_permutations(self) -> int:
        pass


@dataclass
class FixSequenceSampleGen(SampleGen):

    def full_name(self) -> str:
        return f"fix_b{self.branch_factor}_d{self.depth}"

    def _generate_sample(self) -> List[int]:
        root = Node.unflatten_tree_bfs(list(range(self.number_of_nodes)), self.branch_factor)
        return Node.flatten_tree_bfs(root)

    def tree_correct(self, tree_sequence: List[int], error_to_false: bool = True) -> bool:
        return tree_sequence == list(range(self.number_of_nodes))

    @property
    def best_possible_loss(self) -> float:
        return 0

    @property
    def number_of_input_permutations(self) -> int:
        return 1


@dataclass
class SampleLeavesWithReplacementGen(SampleGen):

    def __init__(self, branch_factor: int, depth: int):
        super().__init__(branch_factor, depth)

    def full_name(self) -> str:
        return f"slwr_b{self.branch_factor}_d{self.depth}"

    def _generate_sample(self) -> List[int]:
        tree_sequence = list(range(self.number_of_branching_nodes))
        for i in range(self.number_of_branching_nodes, self.number_of_nodes, self.branch_factor):
            node_id_range = list(range(i, i + self.branch_factor))
            tree_sequence.extend(random.choices(node_id_range, k=self.branch_factor))
        root = Node.unflatten_tree_bfs(tree_sequence, self.branch_factor)
        return Node.flatten_tree_bfs(root)

    def tree_correct(self, tree_sequence: List[int], error_to_false: bool = True) -> bool:
        tree_sequence = list(map(int, list(tree_sequence)))
        try:
            non_leave_sequence = tree_sequence[:self.number_of_branching_nodes]
            leave_sequence = tree_sequence[self.number_of_branching_nodes:]
            correct = non_leave_sequence == list(range(self.number_of_branching_nodes))
            leave_groups = [leave_sequence[i:i + self.branch_factor] for i in range(0, len(leave_sequence), self.branch_factor)]
            leave_value_ranges = [list(range(i, i + self.branch_factor)) for i in range(self.number_of_branching_nodes, self.number_of_nodes, self.branch_factor)]
            for leave_group, leave_values in zip_longest(leave_groups, leave_value_ranges):
                correct &= leave_group is not None and all(leave in leave_values for leave in leave_group)
            return correct
        except Exception as e:
            if not error_to_false:
                raise e
            return False

    @property
    def best_possible_loss(self) -> float:
        seq_length = self.number_of_nodes + 1
        loss_per_siblings_group = -math.log(1 / self.branch_factor) * self.branch_factor
        num_leave_groups = self.number_of_leaves // self.branch_factor
        return loss_per_siblings_group * num_leave_groups / seq_length

    @property
    def number_of_input_permutations(self) -> int:
        leave_group_permutations = self.branch_factor ** self.branch_factor
        number_of_leave_groups = self.number_of_leaves // self.branch_factor
        return leave_group_permutations ** number_of_leave_groups


@dataclass
class SampleNodesWithReplacementGen(SampleGen):

    def __init__(self, branch_factor: int, depth: int):
        super().__init__(branch_factor, depth)

    def full_name(self) -> str:
        return f"snwr_b{self.branch_factor}_d{self.depth}"

    def _generate_sample(self) -> List[int]:
        tree_sequence = [0]
        nodes = deque([0])
        for _ in range(0, (self.number_of_nodes-1) // self.branch_factor):
            node_id = nodes.popleft()
            child_start_id = node_id * self.branch_factor + 1
            node_id_range = list(range(child_start_id, child_start_id + self.branch_factor))
            children = random.choices(node_id_range, k=self.branch_factor)
            tree_sequence.extend(children)
            nodes.extend(children)
        root = Node.unflatten_tree_bfs(tree_sequence, self.branch_factor)
        return Node.flatten_tree_bfs(root)

    def tree_correct(self, tree_sequence: List[int], error_to_false: bool = True) -> bool:
        tree_sequence = list(map(int, list(tree_sequence)))
        try:
            root = Node.unflatten_tree_bfs(tree_sequence, self.branch_factor)
            def check_tree(node, legal_values):
                if node.value not in legal_values:
                    return False
                start_value = node.value * self.branch_factor + 1
                next_legal_values = list(range(start_value, start_value + self.branch_factor))
                for child in node.children:
                    if not check_tree(child, next_legal_values):
                        return False
                return True
            return check_tree(root, [0])
        except Exception as e:
            if not error_to_false:
                raise e
            return False

    @property
    def best_possible_loss(self) -> float:
        seq_length = self.number_of_nodes + 1
        loss_per_siblings_group = -math.log(1 / self.branch_factor) * self.branch_factor
        num_node_groups = (self.number_of_nodes-1) // self.branch_factor
        return loss_per_siblings_group * num_node_groups / seq_length

    @property
    def number_of_input_permutations(self) -> int:
        level = self.depth + 1
        return reduce(lambda a, b: a * b, (self.branch_factor ** (self.branch_factor ** d) for d in range(1, level)))


@dataclass
class SampleLeavesWithoutReplacementGen(SampleGen):

    def __init__(self, branch_factor: int, depth: int):
        super().__init__(branch_factor, depth)

    def full_name(self) -> str:
        return f"slwor_b{self.branch_factor}_d{self.depth}"

    def _generate_sample(self) -> List[int]:
        root = Node.unflatten_tree_bfs(list(range(self.number_of_nodes)), self.branch_factor)
        def shuffle_tree(node):
            if len(node.children) > 1 and len(node.children[0].children) == 0:
                # I am parent of leaves, shuffle my children
                random.shuffle(node.children)
            for child in node.children:
                shuffle_tree(child)
        shuffle_tree(root)
        return Node.flatten_tree_bfs(root)

    def tree_correct(self, tree_sequence: List[int], error_to_false: bool = True) -> bool:
        tree_sequence = list(map(int, list(tree_sequence)))
        try:
            non_leave_sequence = tree_sequence[:self.number_of_branching_nodes]
            leave_sequence = tree_sequence[self.number_of_branching_nodes:]
            correct = non_leave_sequence == list(range(self.number_of_branching_nodes))
            leave_groups = [leave_sequence[i:i + self.branch_factor] for i in range(0, len(leave_sequence), self.branch_factor)]
            leave_value_ranges = [list(range(i, i + self.branch_factor)) for i in range(self.number_of_branching_nodes, self.number_of_nodes, self.branch_factor)]
            for leave_group, leave_values in zip_longest(leave_groups, leave_value_ranges):
                correct &= leave_group is not None and set(leave_group) == set(leave_values)
            return correct
        except Exception as e:
            if not error_to_false:
                raise e
            return False

    @property
    def best_possible_loss(self) -> float:
        seq_length = self.number_of_nodes + 1
        loss_per_siblings_group = sum(-math.log(1 / i) for i in range(1, self.branch_factor + 1))
        num_leaf_groups = self.number_of_leaves // self.branch_factor
        return loss_per_siblings_group * num_leaf_groups / seq_length


    @property
    def number_of_input_permutations(self) -> int:
        leave_group_permutations = math.factorial(self.branch_factor)
        number_of_leave_groups = self.number_of_leaves // self.branch_factor
        return leave_group_permutations ** number_of_leave_groups


@dataclass
class SampleNodesWithoutReplacementGen(SampleGen):

    def __init__(self, branch_factor: int, depth: int):
        super().__init__(branch_factor, depth)

    def full_name(self) -> str:
        return f"snwor_b{self.branch_factor}_d{self.depth}"

    @property
    def best_possible_loss(self) -> float:
        seq_length = self.number_of_nodes + 1
        loss_per_siblings_group = sum(-math.log(1 / i) for i in range(1, self.branch_factor + 1))
        num_node_groups = (self.number_of_nodes - 1) // self.branch_factor
        return loss_per_siblings_group * num_node_groups / seq_length

    @property
    def number_of_input_permutations(self) -> int:
        res = 1
        level = self.depth + 1
        for d in range(1, level):
            res *= math.factorial(self.branch_factor) ** (self.branch_factor ** (d-1))
        return res
        # return reduce(lambda a, b: a * b, (math.factorial(self.branch_factor) * (self.branch_factor ** d // self.branch_factor) for d in range(1, level)))

    def tree_correct(self, tree_sequence, error_to_false: bool=True):
        tree_sequence = list(map(int, list(tree_sequence)))
        try:
            root = Node.unflatten_tree_bfs(tree_sequence, self.branch_factor)
            def sort_tree(node):
                node.children.sort(key=lambda x: x.value)
                for child in node.children:
                    sort_tree(child)
            sort_tree(root)
            return Node.flatten_tree_bfs(root) == list(range(len(tree_sequence)))
        except Exception as e:
            if not error_to_false:
                raise e
            return False

    def _generate_sample(self) -> List[int]:
        root = Node.unflatten_tree_bfs(list(range(self.number_of_nodes)), self.branch_factor)
        def shuffle_tree(node):
            random.shuffle(node.children)
            for child in node.children:
                shuffle_tree(child)
        shuffle_tree(root)
        return Node.flatten_tree_bfs(root)


@dataclass
class XSampleGen(SampleGen):

    def __init__(self, branch_factor: int, depth: int, modulo: int = 2):
        super().__init__(branch_factor, depth)
        self.modulo = modulo
        assert self.branch_factor == 2, 'Only binary trees are supported'

    def full_name(self) -> str:
        return f"x{self.modulo}_b{self.branch_factor}_d{self.depth}"

    @property
    def number_of_input_permutations(self) -> int:
        return -1

    @property
    def best_possible_loss(self) -> float:
        return -1

    def tree_correct(self, tree_sequence, error_to_false: bool=True):
        tree_sequence = list(map(int, list(tree_sequence)))
        try:
            root = Node.unflatten_tree_bfs(tree_sequence, self.branch_factor)
            def un_x_tree(parent: Node, swaps: int):
                if len(parent.children) == 0:
                    return
                was_swap = parent.children[0].value > parent.children[1].value
                if was_swap:
                    # reverse back
                    parent.children.reverse()
                next_swaps = swaps + int(was_swap)
                for child in parent.children:
                    un_x_tree(child, next_swaps)
                # TODO think about % 3, etc. Maybe % 2 provides shortcut?
                if next_swaps != 0 and next_swaps % self.modulo == 0 and len(parent.children) == 2:
                    # reverse back
                    parent.children[0].children, parent.children[1].children = parent.children[1].children, parent.children[0].children
            un_x_tree(root, 0)
            return Node.flatten_tree_bfs(root) == list(range(len(tree_sequence)))
        except Exception as e:
            if not error_to_false:
                raise e
            return False

    def _generate_sample(self) -> List[int]:
        root = Node.unflatten_tree_bfs(list(range(self.number_of_nodes)), self.branch_factor)
        def x_tree(parent: Node, swaps: int):
            swap = random.choice([True, False])
            if swap:
                parent.children.reverse()
            next_swaps = swaps + int(swap)
            for child in parent.children:
                x_tree(child, next_swaps)
            if next_swaps != 0 and next_swaps % self.modulo == 0 and len(parent.children) == 2:
                parent.children[0].children, parent.children[1].children = parent.children[1].children, parent.children[0].children
        x_tree(root, 0)
        return Node.flatten_tree_bfs(root)


@dataclass
class InterleaveSampleGen(SampleGen):

    def __init__(self, sample_gen: SampleGen):
        super().__init__(sample_gen.branch_factor, sample_gen.depth)
        self.sample_gen = sample_gen

    def full_name(self) -> str:
        return f"interleave_{self.sample_gen.full_name()}"

    @property
    def number_of_leaves(self) -> int:
        return self.sample_gen.number_of_leaves * 2

    @property
    def number_of_branching_nodes(self) -> int:
        return self.sample_gen.number_of_branching_nodes * 2

    @property
    def number_of_nodes(self) -> int:
        return self.sample_gen.number_of_nodes * 2

    @property
    def number_of_input_permutations(self) -> int:
        return self.sample_gen.number_of_input_permutations

    @property
    def best_possible_loss(self) -> float:
        seq_length = self.number_of_nodes + 1
        return self.sample_gen.best_possible_loss * 2

    def _generate_sample(self) -> List[int]:
        sample_1 = self.sample_gen.generate_sample()
        sample_2 = self.sample_gen.generate_sample()
        interleaved = [val for pair in zip(sample_1, sample_2) for val in pair]
        return interleaved

    def tree_correct(self, tree_sequence, error_to_false: bool=True):
        tree_sequence = list(map(int, list(tree_sequence)))
        try:
            tree_1 = tree_sequence[::2]
            tree_2 = tree_sequence[1::2]
            return self.sample_gen.tree_correct(tree_1, error_to_false) and self.sample_gen.tree_correct(tree_2, error_to_false)
        except Exception as e:
            if not error_to_false:
                raise e
            return False


class DatasetBosEos(Dataset):

    def __init__(self, length: int, sample_gen: SampleGen, bos_id: int, eos_id: int):
        self.length = length
        self.sample_gen = sample_gen
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        node_ids = self.sample_gen.generate_sample()
        return torch.tensor([self.bos_id] + node_ids), torch.tensor(node_ids + [self.eos_id])


def print_tree(node, level=0, sort=False):
    print(' ' * 4 * level + f'Node({node.value})')
    if sort:
        node.children.sort(key=lambda x: x.value)
    for child in node.children:
        print_tree(child, level + 1, sort=sort)


if __name__ == '__main__':
    BF, D = 5, 2

    gens = [
        # InterleaveSampleGen(FixSequenceSampleGen(branch_factor=BF, depth=D)),
        # XSampleGen(2, 5),
        FixSequenceSampleGen(branch_factor=BF, depth=D),
        SampleLeavesWithReplacementGen(branch_factor=BF, depth=D),
        # SampleNodesWithReplacementGen(branch_factor=BF, depth=D),
        # SampleLeavesWithoutReplacementGen(branch_factor=BF, depth=D),
        # SampleNodesWithoutReplacementGen(branch_factor=BF, depth=D),
    ]
    for gen in gens:
        seq_len = gen.number_of_nodes + 1  # BOS/EOS
        print(f"{gen.full_name()} __ {gen.number_of_input_permutations} __ {gen.best_possible_loss(seq_len)}")
        # print_tree(Node.unflatten_tree_bfs(gen.generate_sample(), gen.branch_factor))

    for gen in gens:
        print(f"Validating {gen.full_name()}")
        for _ in range(1000):
            sample = gen.generate_sample()
            assert gen.tree_correct(sample, error_to_false=False), f"Generated tree is not correct for {gen.full_name()}: {sample}"
    print(f"Validation DONE")

    def generate_samples_chunk(gen, chunk_size):
        return {hashlib.md5(pickle.dumps(array('h', gen.generate_sample()), -1)).hexdigest() for _ in range(chunk_size)}


    def generate_samples_parallel(samples_set, gen, num_samples, num_threads=10):
        chunk_size = num_samples // num_threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(generate_samples_chunk, gen, chunk_size) for _ in range(num_threads)]
            for future in concurrent.futures.as_completed(futures):
                samples_set.update(future.result())


    # try validating number of permutations
    for gen in gens:
        SAMPLES = 1_000_000
        if gen.number_of_input_permutations * 20 > SAMPLES and gen.number_of_input_permutations / 20 < SAMPLES:
            SAMPLES = gen.number_of_input_permutations * 20
        samples = set()
        generate_samples_parallel(samples, gen, SAMPLES)
        assert len(samples) <= gen.number_of_input_permutations, f"{gen.full_name()} failed with {len(samples)} generated but {gen.number_of_input_permutations} expected for {SAMPLES} samples"
        if len(samples) == SAMPLES:
            # Check that number_of_input_permutations is way bigger than SAMPLES
            assert len(samples) < gen.number_of_input_permutations / 10, f"{gen.full_name()} failed as {len(samples)} not equal and not << than {gen.number_of_input_permutations} for {SAMPLES} samples, which is unlikely"
            print(f"{gen.full_name()} maybe with {len(samples)} << {gen.number_of_input_permutations}; used {SAMPLES} samples")
        else:
            assert len(samples) == gen.number_of_input_permutations, f"{gen.full_name()} failed with {len(samples)} generated but {gen.number_of_input_permutations} expected for {SAMPLES} samples"
            print(f"{gen.full_name()} passed with {len(samples)} == {gen.number_of_input_permutations}; used {SAMPLES} samples")
