import bz2
import math
import os
from scipy.fftpack import dct, idct
from PIL import Image
import numpy as np
import huffman
import pickle


def dct2(b):
    return dct(dct(b.T, norm='ortho').T, norm='ortho')


block_size = 8
QUANTIZATION_MAT_LUM = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])

QUANTIZATION_MAT_Chr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                [18, 21, 26, 66, 99, 99, 99, 99],
                                [24, 26, 56, 99, 99, 99, 99, 99],
                                [47, 66, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99]])


filename = "photo1.png"
rgb = Image.open(filename)
width, height = rgb.size

#####convert from RGB to YCbCr
ycbcr = rgb.convert("YCbCr")
y, cb, cr = ycbcr.split()
y = np.array(y)
cb = np.array(cb)
cr = np.array(cr)

#####subsampling 4:2:0
if height % 2 != 0:
    cb[1:height - 2:2, :] = cb[:height - 3:2, :]
    cr[1:height - 2:2, :] = cr[:height - 3:2, :]
else:
    cb[1::2, :] = cb[::2, :]
    cr[1::2, :] = cr[::2, :]
if width % 2 == 0:
    cb[:, 1::2] = cb[:, ::2]
    cr[:, 1::2] = cr[:, ::2]
else:
    cb[:, 1:width - 2:2] = cb[:, :width - 3:2]
    cr[:, 1:width - 2:2] = cr[:, :width - 3:2]

merged_ycbcr = np.dstack((y, cb, cr))

nbh = math.ceil(height/block_size)
nbw = math.ceil(width/block_size)
H = block_size * nbh
W = block_size * nbw
######for filling incomplet blocks with black color
block_img = np.full((H, W, 3), [16, 128, 128])
dct_mat = []
for row in np.arange(H - block_size + 1, step=block_size):
    for col in np.arange(W - block_size + 1, step=block_size):
        if row + block_size < height and col + block_size < width:
            #####divide to 8x8 blocks
            block_img[row : row + block_size, col : col + block_size] = merged_ycbcr[row : row + block_size, col : col + block_size]
        block = block_img[row : (row + block_size), col : (col + block_size)]
        #####calculate DCT of each block
        dct_y = dct2(block)[..., 0]
        dct_cb = dct2(block)[..., 1]
        dct_cr = dct2(block)[..., 2]
        #####quantization of DCT coefficients
        dct_y_q = np.divide(dct_y, QUANTIZATION_MAT_LUM).astype(int)
        dct_cb_q = np.divide(dct_cb, QUANTIZATION_MAT_Chr).astype(int)
        dct_cr_q = np.divide(dct_cr, QUANTIZATION_MAT_Chr).astype(int)
        dct_q_merged = np.dstack((dct_y_q, dct_cb_q, dct_cr_q))
        dct_mat.append(dct_q_merged)

print("required bits for original image", height*width*24)
#####apply huffman coding to quantized DCT coefficients
encoding, tree = huffman.huffman_encoding(dct_mat)

save_objects = [encoding, tree]
compress_file = bz2.BZ2File("compress", "wb")
pickle.dump(save_objects, compress_file)
compress_file.close()
print("size of saved file: ", os.path.getsize("compress"))