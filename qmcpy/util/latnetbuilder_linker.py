from ..util import NotYetImplemented
import numpy as np


def latnetbuilder_linker(lnb_dir="./", out_dir="./", fout_prefix="lnb4qmcpy"):
    """
    Args:
        lnb_dir (str): relative path to directory where `outputMachine.txt` is
            stored e.g. 'my_lnb/poly_lat/'
        out_dir (str): relative path to directory where output should be stored
            e.g. 'my_lnb/poly_lat_qmcpy/'
        fout_prefix (str): start of output file name. e.g. 'my_poly_lat_vec'

    Returns:
        path to file which can be passed into QMCPy's Lattice or Sobol' in
        order to use the linked latnetbuilder generating vector/matrix e.g.
        'my_poly_lat_vec.10.16.npy'

    Adapted from latnetbuilder parser:
        https://github.com/umontreal-simul/latnetbuilder/blob/master/python-wrapper/latnetbuilder/parse_output.py#L74
    """
    raise NotYetImplemented("latnetbuilder_linker is currently under reconstruction.")
    with open(lnb_dir + "/outputMachine.txt") as f:
        Lines = f.read().split("\n")
    sep = "  //"
    if Lines[0].split(sep)[0] == "Ordinary":
        nb_points = int(Lines[1].split(sep)[0])
        dim = int(Lines[2].split(sep)[0])
        gen_vector = []
        for i in range(dim):
            gen_vector.append(int(Lines[5 + i].split(sep)[0]))
        v = np.array(gen_vector, dtype=np.uint64)
        f_out = "%s/%s.%d.%d.npy" % (out_dir, fout_prefix, dim, np.log2(nb_points))
        #save(f_out, v)
        return f_out
    else:
        nb_cols = int(Lines[0].split(sep)[0])
        nb_rows = int(Lines[1].split(sep)[0])
        nb_points = int(Lines[2].split(sep)[0])
        dim = int(Lines[3].split(sep)[0])
        set_type = Lines[5].split(sep)[0]
        line = 6
        if set_type == "Polynomial":
            line += dim + 1
        elif set_type == "Sobol":
            line += dim
        mint = []
        pows2 = 2 ** np.arange(nb_rows - 1, -1, -1)[:, None]
        for c in range(dim):
            line += 1
            M = []
            for i in range(nb_rows):
                M.append(np.array([int(x) for x in Lines[line + i].split(" ")]))
            line += nb_rows
            mint.append((np.array(M) * pows2).sum(0))
        mint = np.array(mint, dtype=np.uint64)
        f_out = "%s/%s.%d.%d.%d.msb.npy" % (out_dir, fout_prefix, dim, nb_rows, nb_cols)
        #save(f_out, mint)
        return f_out
