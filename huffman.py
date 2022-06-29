import numpy as np
from numpy import copy


class Node:
    def __init__(self, freq, tag, left=None, right=None):
        self.freq = freq
        self.tag = tag
        self.left = left
        self.right = right
        self.code = ''

    def __str__(self):
        return "node {0}: p = {1}, left = [{2}, right = {3}]".format(self.tag, self.freq, self.left, self.right)


codes = dict()
def calculate_codes(node, val=''):
    if node.left:
        calculate_codes(node.left, val + str(node.code))
    if node.right:
        calculate_codes(node.right, val + str(node.code))
    if not node.left and not node.right:
        codes[node.tag] = val + str(node.code)
    return codes


def calculate_freq(data):
    unique, counts = np.unique(data, return_counts=True)
    occur_count_dict = dict(zip(unique, counts))
    return occur_count_dict


def encode_res(data, coding):
    encoding_output = copy(data).astype(str)
    for k, v in coding.items():
        encoding_output[data==k] = v
    return encoding_output


def compression_ratio(calculate_freq_dict, coding):
    before_compression = 0
    for k, v in calculate_freq_dict.items():
        before_compression += v
    before_compression *= 8
    after_compression = 0
    for symbol in coding.keys():
        count = calculate_freq_dict[symbol]
        after_compression += count * len(coding[symbol])
    print("required bits before coding: ", before_compression)
    print("required bits after coding: ", after_compression)
    print("Compression ratio by huffman coding: {:.2f}".format(((before_compression - after_compression)/before_compression)*100))


def huffman_encoding(data):
    calculate_freq_dict = calculate_freq(data)
    nodes = []
    for symbol in calculate_freq_dict.keys():
        nodes.append(Node(calculate_freq_dict[symbol], symbol))

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)
        right = nodes[0]
        left = nodes[1]
        left.code = 0
        right.code = 1
        newNode = Node(left.freq + right.freq, left.tag + right.tag, left, right)
        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = calculate_codes(nodes[0])
    print("coding: ", huffman_encoding)
    encoded_output = encode_res(data, huffman_encoding)
    compression_ratio(calculate_freq_dict, huffman_encoding)
    return encoded_output, nodes[0]


'''
data = [8, 8, 34, 5, 10, 34, 6, 43, 127, 10, 10, 8, 10, 34, 10]
print(data)
encoding, tree = huffman_encoding(data)
print("Encoded output", encoding)
'''