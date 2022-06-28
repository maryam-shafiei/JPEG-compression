import numpy as np
from numpy import copy


class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right

        # tree direction (0/1)
        self.code = ''

    def __str__(self):
        return "node {0}: p = {1}, left = [{2}, right = {3}]".format(self.symbol, self.prob, self.left, self.right)


""" A helper function to print the codes of symbols by traveling Huffman Tree"""
codes = dict()

###################################################???????????????????
def Calculate_Codes(node, val=''):
    newVal = val + str(node.code)
    if (node.left):
        Calculate_Codes(node.left, newVal)
    if (node.right):
        Calculate_Codes(node.right, newVal)

    if (not node.left and not node.right):
        codes[node.symbol] = newVal

    return codes


""" A helper function to calculate the probabilities of symbols in given data"""


def Calculate_Probability(data):
    unique, counts = np.unique(data, return_counts=True)
    occur_count_dict = dict(zip(unique, counts))
    return occur_count_dict


""" A helper function to obtain the encoded output"""


def Output_Encoded(data, coding):
    encoding_output = copy(data).astype(str)
    for k, v in coding.items():
        encoding_output[data==k] = int(v)
    return encoding_output


""" A helper function to calculate the space difference between compressed and non compressed data"""


def Total_Gain(calculate_probability_dict, coding):
    before_compression = 0
    for k, v in calculate_probability_dict.items():
        before_compression += v
    before_compression *= 8  # total bit space to stor the data before compression
    after_compression = 0
    for symbol in coding.keys():
        count = calculate_probability_dict[symbol]
        after_compression += count * len(coding[symbol])  # calculate how many bit is required for that symbol in total
    #before_compression /= (2**23)
    #after_compression /= (2**23)
    print("Space usage before compression (in bits):", before_compression)
    print("Space usage after compression (in bits):", after_compression)


def Huffman_Encoding(data):
    calculate_probability_dict = Calculate_Probability(data)
    print("symbols: ", calculate_probability_dict.keys())
    print("probabilities: ", calculate_probability_dict.values())

    nodes = []

    # converting symbols and probabilities into huffman tree nodes
    for symbol in calculate_probability_dict.keys():
        nodes.append(Node(calculate_probability_dict[symbol], symbol))

    while len(nodes) > 1:
        # sort all the nodes in ascending order based on their probability
        nodes = sorted(nodes, key=lambda x: x.prob)

        # pick 2 smallest nodes
        right = nodes[0]
        left = nodes[1]

        left.code = 0
        right.code = 1

        # combine the 2 smallest nodes to create new node
        newNode = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = Calculate_Codes(nodes[0])
    print("symbols with codes", huffman_encoding)
    Total_Gain(calculate_probability_dict, huffman_encoding)
    encoded_output = Output_Encoded(data, huffman_encoding)
    return encoded_output, nodes[0]


""" First Test """
#data = [8, 8, 34, 5, 10, 34, 6, 43, 127, 10, 10, 8, 10, 34, 10]
#print(data)
#encoding, tree = Huffman_Encoding(data)
#print("Encoded output", encoding)
