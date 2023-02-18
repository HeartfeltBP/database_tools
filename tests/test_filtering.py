import pytest
from database_tools.io.records import get_header_record
from database_tools.filtering.exclusion import header_has_signals


@pytest.mark.parametrize('input,expected', [(('30/3000063',               'layout', ['PLETH', 'ABP']),  True),
                                            (('30/3000063',               'layout', ['PLETH', 'FAIL']), False),
                                            (('30/3000063/3000063_0016',  'data',   ['PLETH', 'ABP']),  True)])
def test_header_has_signals(input, expected):
    hea = get_header_record(path=input[0], record_type=input[1])
    result = header_has_signals(hea, signals=input[2])
    assert result == expected
