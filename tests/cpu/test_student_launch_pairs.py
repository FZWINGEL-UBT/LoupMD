import pytest

from tools.dev.student_launch_pairs import compare_pairs


def test_pair_requires_both_orders_and_launch_reduction():
    a, b = 'separate', 'grouped'
    assert compare_pairs([a, b, b, a], [200, 190, 180, 200], 57, 37)['material_in_both_orders']
    assert not compare_pairs([a, b, b, a], [200, 190, 199, 200], 57, 37)['material_in_both_orders']
    with pytest.raises(ValueError):
        compare_pairs([a, b, a, b], [200, 190, 200, 190], 57, 37)
    with pytest.raises(ValueError):
        compare_pairs([a, b, b, a], [200, 190, 180, 200], 57, 57)
