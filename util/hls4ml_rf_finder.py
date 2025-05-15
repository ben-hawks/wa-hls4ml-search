from bisect import bisect_left
import math

# Functions stolen from hls4ml/backends/fpga/fpga_backend.py to get closest valid reuse factor for a given n_in & n_out


def get_valid_reuse_factors(n_in, n_out):
    max_rf = n_in * n_out
    valid_reuse_factors = []
    for rf in range(1, max_rf + 1):
        _assert = _validate_reuse_factor(n_in, n_out, rf)
        if _assert:
            valid_reuse_factors.append(rf)
    return valid_reuse_factors


def _validate_reuse_factor(n_in, n_out, rf):
    multfactor = min(n_in, rf)
    multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
    #
    # THIS ASSERTION IS FOR THE FUNCTIONAL CORRECTNESS OF THE DENSE LAYER
    #
    _assert = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
    _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
    #
    # THIS ASSERTION IS FOR QoR AND EXECUTION TIME
    #
    _assert = _assert and (((n_in * n_out) % rf) == 0)

    return _assert


def get_closest_reuse_factor(valid_rf, chosen_rf):
    """
    Returns closest value to chosen_rf. valid_rf is sorted (obtained from get_valid_reuse_factors())
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(valid_rf, chosen_rf)
    if pos == 0:
        return valid_rf[0]
    if pos == len(valid_rf):
        return valid_rf[-1]
    before = valid_rf[pos - 1]
    after = valid_rf[pos]
    if after - chosen_rf < chosen_rf - before:
        return after
    else:
        return before


def set_closest_reuse_factor(chosen_rf, n_in, n_out, attribute='reuse_factor'):
    assert attribute is not None, 'Reuse factor attribute cannot be None'
    valid_rf = get_valid_reuse_factors(n_in, n_out)
    if chosen_rf not in valid_rf:
        closest_rf = get_closest_reuse_factor(valid_rf, chosen_rf)
        valid_rf_str = ','.join(map(str, valid_rf))
        print(
            f'WARNING: Invalid ReuseFactor={chosen_rf} with layer parameters n_in={n_in} and n_out={n_out}.'
            f'Using ReuseFactor={closest_rf} instead. Valid ReuseFactor(s): {valid_rf_str}.'
        )
        return closest_rf
    else:
        return chosen_rf
