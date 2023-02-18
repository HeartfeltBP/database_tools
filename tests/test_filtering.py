import pytest
from database_tools.io.records import get_header_record
from database_tools.filtering.exclusion import layout_has_signals


@pytest.mark.parametrize('input,expected', [(['PLETH', 'ABP'], True),
                                            (['PLETH', 'FAIL'], False)])
def test_layout_has_signals(input, expected):
    hea = get_header_record(path='30/3000063', record_type='layout')
    result = layout_has_signals(hea, signals=input)
    assert result == expected
