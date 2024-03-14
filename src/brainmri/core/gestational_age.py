import numpy as np

from brainmri.constants import GESTATIONAL_AGE_IDS, GESTATIONAL_AGES


def gestational_age(id_nr: int) -> float | None:
    """Get gestational age from id number.

    Args:
        id_nr (int): ID number

    Returns:
        float | None: Gestational age in weeks or None if not found
    """
    if id_nr in GESTATIONAL_AGE_IDS:
        idx = GESTATIONAL_AGE_IDS.index(id_nr)
        ga = GESTATIONAL_AGES[idx]

        if (type(ga) == int or type(ga) == float) and not np.isnan(ga):
            ga = float(ga)
        else:
            ga = None
    else:
        ga = None
    return ga
