import wfdb
import pytest
import numpy as np
from database_tools.io.records import generate_record_paths, get_data_record, get_header_record, header_has_signals, get_signal

NoneType = type(None)


@pytest.mark.parametrize('input,expected', [(None,       list),
                                            ('adults',   list),
                                            ('neonates', list)])
def test_generate_record_paths(input, expected):
    result = [x for x in generate_record_paths(name=input)]
    assert isinstance(result, expected)

@pytest.mark.parametrize('input,expected', [(('30/3000063',              'layout'), wfdb.Record),
                                            (('30/3000303',              'layout'), NoneType),
                                            (('30/3000063',              'data'),   wfdb.MultiRecord),
                                            (('30/3000063/3000063_0016', 'data'),   wfdb.Record)])
def test_get_header_record(input, expected):
    result = get_header_record(path=input[0], record_type=input[1])
    assert isinstance(result, expected)

@pytest.mark.parametrize('input,expected', [(('30/3000063',               'layout', ['PLETH', 'ABP']),  True),
                                            (('30/3000063',               'layout', ['PLETH', 'FAIL']), False),
                                            (('30/3000063/3000063_0016',  'data',   ['PLETH', 'ABP']),  True)])
def test_header_has_signals(input, expected):
    hea = get_header_record(path=input[0], record_type=input[1])
    result = header_has_signals(hea, signals=input[2])
    assert result == expected

@pytest.mark.parametrize('input,expected', [('30/3000063/3000063_0016', wfdb.Record),
                                            ('30/3000003/3000063_0016', NoneType)])
def test_get_data_record(input, expected):
    result = get_data_record(path=input)
    assert isinstance(result, expected)

@pytest.mark.parametrize('input,expected,exception', [(('30/3000063/3000063_0016', 'PLETH'), np.ndarray, None),
                                                      (('30/3000003/3000063_0016', 'FAIL'),  None,       ValueError)])
def test_get_signal(input, expected, exception):
    rcd = get_data_record(path=input[0])
    if expected is not None:
        assert isinstance(get_signal(rec=rcd, sig=input[1], errors='ignore'), np.ndarray)
    else:
        with pytest.raises(exception):
            get_signal(rec=rcd, sig=input[1], errors='raise')
