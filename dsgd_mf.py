import sys, random
import numpy as np
from scipy.sparse import csr_matrix
from pyspark import SparkContext, SparkConf

def load_CSV_file():
    """
        Load CSV file and construct sparse matrix V.
    """
    row = []
    col = []
    data = []

    lines = sc.textFile(inputV_filepath).collect()
    for line in lines:
        line_arr = line.split(",")
        row.append(int(line_arr[0]) - 1)
        col.append(int(line_arr[1]) - 1)
        data.append(float(line_arr[2]))

    return csr_matrix((data, (row, col)))

def factorize_matrix():
    """
        Factorize matrix V into two matrices and return them as a tuple.
    """
    # initialize W and H
    num_users, num_movies = V.get_shape()
    W = np.random.rand(num_users, num_factors)
    H = np.random.rand(num_factors, num_movies)

    # get the size of each block
    block_row_size = num_users / num_workers
    block_col_size = num_movies / num_workers

    # while not converged
    for t in range(num_iterations):
        # select a stratum S = {(I1, J1), ..., (IB, JB)} of B blocks
        for stratum in range(num_workers):
            blocks = []
            col_offset = block_col_size * stratum
            for row in range(num_workers):
                # compute the border
                top = row * block_row_size
                left = col_offset
                bottom = (row + 1) * block_row_size
                if bottom > num_users:
                    bottom = num_users
                right = col_offset + block_col_size
                if right > num_movies:
                    right = num_movies
                # construct and add a new block of the stratum
                blocks.append((V[top:bottom, left:right],
                               W[top:bottom, :],
                               H[:, left:right],
                               top, bottom, left, right))
                # if move to the right-most border, update column offset
                col_offset = right
                if col_offset == num_movies:
                    col_offset = 0

            # do partition and perform sequential SGD on each block
            n = t * num_workers + stratum
            results = sc.parallelize(blocks, num_workers) \
                        .map(lambda x: do_SGD(x, n)) \
                        .collect()

            # collect the updated parameter blocks from all workers and update W and H
            for top, bottom, left, right, W_res, H_res in results:
                W[top:bottom, :] = W_res
                H[:, left:right] = H_res

        # print L_NZSL(V, W, H)
    return (W, H)

def do_SGD(block, n):
    """
        Given a block, perform sequential SGD on it.
    """
    V, W, H, top, bottom, left, right = block
    # get all the non-zero indices
    non_zero_rows, non_zero_cols = V.nonzero()

    for i in range(100):
        # randomly select a sample
        k = random.randint(0, len(non_zero_rows) - 1)
        i = non_zero_rows[k]
        j = non_zero_cols[k]

        grad_W = -2 * (V[i, j] - W[i, :].dot(H[:, j])) * H[:, j] \
               + 2 * lambda_value / len(V[i, :].nonzero()[0]) * W[i, :].transpose()
        grad_H = -2 * (V[i, j] - W[i, :].dot(H[:, j])) * W[i, :].transpose() \
               + 2 * lambda_value / len(V[i, :].nonzero()[0]) * H[:, j]

        epsilon = (500 + n + i) ** (-beta_value)
        W[i, :] -= grad_W * epsilon
        H[:, j] -= grad_H * epsilon

    return (top, bottom, left, right, W, H)

def L_NZSL(V, W, H):
    """
        Reconstruct V by W and H and compute the non-zero squared sum.
    """
    V_hat = W.dot(H)
    nonzero_idx = V.nonzero()
    diff = np.asarray(V[nonzero_idx] - V_hat[nonzero_idx])
    return np.sum(diff ** 2)

if __name__ == "__main__":
    """
        Usage: spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations>
        <beta_value> <lambda_value>
        <inputV_filepath> <outputW_filepath> <outputH_filepath>
    """
    if len(sys.argv) < 9:
        print "[Error] Not enough parameters. See usage."
        exit(1)

    # read in parameters
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]

    # initialize the SparkContext
    conf = SparkConf().setAppName("DSGD_MF")
    sc = SparkContext(conf=conf)

    # read in V
    V = load_CSV_file()

    # do DSGD Matrix Factorization
    W, H = factorize_matrix()

    # output the result
    np.savetxt(outputW_filepath, W, delimiter=",", fmt="%.12f")
    np.savetxt(outputH_filepath, H, delimiter=",", fmt="%.12f")

    # close the SparkContext
    sc.stop()
