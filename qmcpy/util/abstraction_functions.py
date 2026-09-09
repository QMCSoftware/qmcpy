import numpy as np
from copy import copy


def _univ_repr(qmc_object: object, abc_class_name: str, attributes: list):
    """Clean way to represent qmc_object data.

    Args:
        qmc_object (object): an qmc_object instance
        abc_class_name (str): name of the abstract class
        attributes (list): list of attributes to include

    Returns:
        str: string representation of this qmcpy object

    Notes:
        print(qmc_object) is equivalent to print(qmc_object.__repr__()). See an
        abstract classes __repr__ method for example call to this method.
    """
    with np.printoptions(precision=3, threshold=10):
        unique_attributes = []
        for attrib in attributes:
            if attrib not in unique_attributes:
                unique_attributes += [attrib]
        string = "%s (%s)" % (type(qmc_object).__name__, abc_class_name)
        for key in unique_attributes:
            val = getattr(qmc_object, key)
            # list of one value becomes just that value
            if isinstance(val, list) and len(val) == 1:
                val = copy(val[0])
            elif isinstance(val, list):
                try:
                    val = np.array(val).copy()
                except Exception:
                    pass
            elif isinstance(val, np.ndarray):
                val = val.copy().squeeze()
                if val.shape == ():
                    val = val.item()
                elif val.size == 1:
                    val = val[0].item()
            # printing options
            s = "    %-15s " % key
            if isinstance(val, int) or isinstance(val, float):  # scalar
                p = 0.1 if val <= 0 else np.log2(float(val))
                if (p % 1 == 0) and p != 0:  # power of 2
                    s += "2^(%d)" % int(p)
                elif isinstance(val, int) or (val % 1 == 0):  # int
                    s += "%d" % int(val)
                else:  # float
                    if abs(val) < 0.001:  # exponential format
                        s += "%.2e" % val
                    else:  # float format
                        s += "%.3f" % val
            else:
                s += "%s" % str(val)
            string += "\n" + s.replace("\n", "\n     %-15s" % " ")
    return string
