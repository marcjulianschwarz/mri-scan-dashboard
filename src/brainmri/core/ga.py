import numpy as np

from brainmri.constants import gas, gas_ids


def gestational_age(id_nr: int):
    if id_nr in gas_ids:
        idx = gas_ids.index(id_nr)
        ga = gas[idx]

        if (type(ga) == int or type(ga) == float) and not np.isnan(ga):
            ga = float(ga)
        else:
            ga = None
    else:
        ga = None
    return ga
