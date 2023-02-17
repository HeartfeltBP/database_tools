import pytest
import wfdb
import numpy as np
from database_tools.io.records import get_data_record, get_layout_record, get_signal

NoneType = type(None)

@pytest.mark.parametrize('input,expected', [('30/3000063', wfdb.Record),
                                            ('30/3000303', NoneType)])
def test_get_layout_record(input, expected):
    result = get_layout_record(path=input)
    assert isinstance(result, expected)

@pytest.mark.parametrize('input,expected', [('30/3000063/3000063_0016', wfdb.Record),
                                            ('30/3000003/3000063_0016', NoneType)])
def test_get_data_record(input, expected):
    result = get_data_record(path=input)
    assert isinstance(result, expected)

@pytest.mark.parametrize('input,expected,exception', [(('30/3000063/3000063_0016', 'PLETH'), np.ndarray, None),
                                                      (('30/3000003/3000063_0016', 'III'),   None,       ValueError)])
def test_get_signal(input, expected, exception):
    rcd = get_data_record(path=input[0])
    if expected is not None:
        result = get_signal(rec=rcd, sig=input[1], errors='ignore')
    else:
        with pytest.raises(exception):
            get_signal(rec=rcd, sig=input[1], errors='raise')
