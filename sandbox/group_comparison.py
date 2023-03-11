import numpy as np
from typing import Callable
from tabulate import tabulate
from sandbox.fairness_metrics import test_independence, test_separation, test_sufficiency
from functools import partial
from scipy import sparse

classifier = ...
X = ...
y = ...


# define diagnostic system
headers=['Group A', 'Group B', 'Test conclusion']
combinations = [
    ('V1_sex_male', 'V1_sex_female'),
    ('V8_age_14', 'V8_age_15'),
    ('V8_age_16', 'V8_age_17'),
    ('V4_area_origin_Europe', 'V4_area_origin_Latin America'),
    ('V4_area_origin_Maghreb', 'V4_area_origin_Other')
]
independence = partial(test_independence, classifier, X)
separation = partial(test_seperation, X, y)
separation = partial(test_sufficiency, X, y)

test_signature = Callable[[], np.ndarray]
def tabulate_diagnostic(group_combinations: np.ndarray, headers:list[str], test_fn: test_signature):
    result = np.hstack((group_combinations, test_fn()))
    return tabulate(result, headers, tablefmt='latex')

# compute results by evaluating the system
tabulate_diagnostic(combinations, headers, independence)
tabulate_diagnostic(combinations, headers, separation)
tabulate_diagnostic(combinations, headers, sufficiency)