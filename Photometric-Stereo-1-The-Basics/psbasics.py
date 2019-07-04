# Copyright 2019 Ian Hales
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
import numpy as np
from scipy import misc
from matplotlib import pyplot as plt


def sanitise_image(image):
    return (image / 255).flatten()


def ps_3_source(images, L):
    """Images should be a 3 element list of single-channel greyscale images.
    L is a 3x3 numpy array containing the light direction vectors, with each row being one light vector"""

    # We'll need this later...
    original_size = images[0].shape[:2]

    # Normalise all images and convert to row vectors (each image is one row)
    images = np.vstack(map(sanitise_image, images))

    # Make sure that lighting vectors are normalised
    L = L / np.linalg.norm(L, ord=2, axis=1, keepdims=True)

    # Solve for G = N'*rho using Normal equations
    norm_sln = np.linalg.inv(L.T * L) * L.T

    # We can do this easily for each pixel. norm_sln is 3x3, images is 3xn (where n i num pixels)
    G = np.matmul(norm_sln, images)

    # The albedo is just the column-wise L2 norm (magnitude) of G...
    rho = np.linalg.norm(G, axis=0)

    # The normal map is simply each column of G divided the equivalent column in the albedo
    N = np.divide(G, np.vstack([rho] * 3))

    # Reshape back to image
    rho = rho.reshape(original_size)

    # We need to transpose the normal list before we reshape
    N = N.T.reshape(original_size[0], original_size[1], 3)

    return N, rho


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("USAGE: psbasics <lights_filename> <image_1_filename> <image_3_filename> <image_3_filename>")
        sys.exit()

    L = np.loadtxt(sys.argv[1])

    # Create a list from the input images
    images = []
    for i in range(3):
        images.append(misc.imread(sys.argv[i + 2]))

    # Run the PS algorithm
    N, rho = ps_3_source(images, L)

    plt.subplot('121')
    plt.imshow(N)
    plt.subplot('122')
    plt.imshow(255 * rho, cmap='gray')
    plt.show()
