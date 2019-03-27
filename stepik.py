import pandas as pd
import re
import numpy as np
from sympy import lambdify
from sympy import Symbol
from math import isinf

# # %% number one
# input = pd.read_csv('/Users/zagidull/Downloads/input.txt', sep=' ', skiprows=1, header=None)
# out = input.sum(axis=1)
# out.to_csv('/Users/zagidull/Downloads/output.txt', index=False, header=False)
#
# # %% number two
# input = pd.read_csv('/Users/zagidull/Downloads/input 2.txt', skiprows=1, header=None, squeeze=True)
# holder = []
# for i in range(19):
#     if i % 2 == 0:
#         haystack = input[i]
#         needle = input[i + 1]
#         a = [m.start() for m in re.finditer(r'(?={})'.format(re.escape(needle)), haystack)]  # search with the lookback
#         a = [x + 1 for x in a]  # adjusting for index starting at 1
#         a = re.sub(',', ' ', str(a))  # remove comma from the list after making a into a string
#         a = re.sub(']', ' ', str(a))  # remove parenthesis
#         a = re.sub('\[+', ' ', str(a))
#         print(a)
#         holder.append(str(a))
#
# # holder = [[number+1 for number in group] for group in holder]
#
# with open("/Users/zagidull/Downloads/output 2.txt", "w") as f:
#     for el in holder:
#         f.write('%s\n' % el)
# %% number three
# ToDo a fucking closed-form solution!!!

input = pd.read_csv('/Users/zagidull/Downloads/tests.txt', skiprows=1, header=None, sep=' ')
input.rename(index=str, columns={0: "n", 1: 'a', 2: "b"}, inplace=True)


# %%
# def err_handler(type, flag):
#     print("Floating point error (%s), with flag %s" % (type, flag))
#
#
# saved_handler = np.seterrcall(err_handler)
# save_err = np.seterr(all='call')


def calc1(n, a, b, loops=10, ret=False):
    holder = []
    z = float(n)
    n = Symbol('n')
    expr = a * n - b * (n ** 2)
    f = lambdify(n, expr)
    data = np.array([f(z)], dtype=np.float64)  # calculate the very first iteration
    data = np.append(data, f(data[-1]))  # calculate second iteration
    i = 0
    while i < loops:
        if f(data[-1]) < -0.1:  # this is a check if we get negative mass of bees. That is not possible
            return 0
        else:
            if isinf(f(data[-1])):  # this is to check if we reached infinity, ie overflow handler
                print(n, a, b, i)
                return -1
            elif f(data[-1]) > 1000000000000000:  # this is also a lazy handler for overflows
                print(n, a, b, i)
                return -1
            else:
                z = 0
                while z < 10000:  # this is actual numerical calculations. Here 1000 or even 100 is enough
                    data = np.append(data, f(data[-1]))
                    z = z + 1
        #         print(z)
        # print(i)
        i = i + 1
    if ret:

        holder.append(data)
        return np.mean(data[-2:]), holder
    else:
        return np.mean(data[-2:])  # get mean of last two values in case it is an oscillating function
    # i = i + 1


#    return np.mean(data[-2:])

input['result'] = input.apply(lambda x: calc1(x.values[0], x.values[1], x.values[2]), axis=1)
input['result'].to_csv('result3.txt', sep=' ', header=False, index=False)

# %% problem 2.3a and 2.3b
'''
In this version the sequencing machine only makes A->T, T->A, G->C and C->G errors.
https://stepik.org/lesson/201239/step/2?unit=179663

'''
import itertools

input = pd.read_csv('/Users/zagidull/Downloads/1.txt', skiprows=1, header=None, sep=' ')
input.rename(index=str, columns={0: "length_genome", 1: 'length_read', 2: "p_error", 3: "num_reads"}, inplace=True)


# %%
def bla(starting_position, length_read, length_genome):
    if length_genome < starting_position + length_read:
        return starting_position + length_read - length_genome
    elif length_genome == starting_position + length_read:
        return length_genome
    else:
        return starting_position + length_read


# %%
def prob(length_genome, length_read, num_reads):
    holder = np.zeros((length_genome ** num_reads, num_reads), dtype=np.int16)
    #    counter = [list(p) for p in itertools.product(range(length_genome), repeat=num_reads)]
    for i, p in enumerate(itertools.product(range(length_genome), repeat=num_reads)):
        holder[i] = p
    out = []
    out_unread = []
    for x in holder:
        out_inner = []
        for start in x:
            end = bla(start, length_read, length_genome)
            if start > end:
                the_whole = []
                z = list(range(length_genome))
                for i in itertools.count(start, 1):
                    the_whole.append(z[i % len(z)])
                    if len(the_whole) == length_read:
                        out_inner.append(the_whole)
                        break
            else:
                the_whole = list(range(start, end))
                out_inner.append(the_whole)
        out_inner = [item for sublist in out_inner for item in sublist]
        #        print(out_inner)
        unread = set(range(length_genome)) - set(out_inner)
        out_unread.append(unread)
        lengths = []
        for x in out_unread:
            lengths.append(len(x))
        unique_lengths = set(lengths)
        dic = {}
        for x in unique_lengths:
            dic[x] = lengths.count(x) / len(lengths)
        final = []
        for key, value in dic.items():
            final.append(value * key * 0.75)
        out.append(out_inner)
    return holder, sum(final)


# %%
def prob1(length_genome, length_read, num_reads):
    x = np.zeros((length_genome ** length_read, length_read), dtype=np.int16)
    # counter = [list(p) for p in itertools.product(range(length_genome), repeat=num_reads)]
    for i, p in enumerate(itertools.product(range(length_genome), repeat=length_read)):
        x[i] = p
    return (x)


# %%
def prob2(length_genome, length_read, num_reads):
    #    counter = [list(p) for p in itertools.product(range(length_genome), repeat=num_reads)]
    def qq(x):
        for start in x:
            end = bla(start, length_read, length_genome)
            if start > end:
                inner = np.zeros((2,length_read))
                for i,v in enumerate(itertools.count(start, 1)):
                    inner[i] = v
                    if i == length_read:
                        break
                    return z[i % len(z)]
            else:
                return range(start, end)

    holder = np.zeros((length_genome ** num_reads, num_reads), dtype=np.int16)
    z = list(range(length_genome))
    for i, p in enumerate(itertools.product(range(length_genome), repeat=num_reads)):
        holder[i] = qq(p)

    return holder
    # %%

    out = []
    out_unread = []
    for x in holder:
        out_inner = []
        for start in x:
            end = bla(start, length_read, length_genome)
            if start > end:
                the_whole = []
                z = list(range(length_genome))
                for i in itertools.count(start, 1):
                    the_whole.append(z[i % len(z)])
                    if len(the_whole) == length_read:
                        out_inner.append(the_whole)
                        break
            else:
                the_whole = list(range(start, end))
                out_inner.append(the_whole)
        out_inner = [item for sublist in out_inner for item in sublist]
        #        print(out_inner)
        unread = set(range(length_genome)) - set(out_inner)
        out_unread.append(unread)
        lengths = []
        for x in out_unread:
            lengths.append(len(x))
        unique_lengths = set(lengths)
        dic = {}
        for x in unique_lengths:
            dic[x] = lengths.count(x) / len(lengths)
        final = []
        for key, value in dic.items():
            final.append(value * key * 0.75)
        out.append(out_inner)
    return holder, sum(final)


# %%
z = 0
m = []
lister = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in itertools.count(6, 1):
    z = z + 1
    list_element = lister[i % len(lister)]
    m.append(list_element)
    if z > 10:
        break
print(m)

# %%
xvalues = np.array(range(4))  # [0 1 2 3]
yvalues = np.array(range(4))
out = [[x0, y0] for x0 in xvalues for y0 in yvalues]


# %%
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out
