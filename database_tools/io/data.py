def write_to_json(data: str, file_path: str) -> None:
    """Writes data to .json or .jsonlines file. 

    Args:
        data (str): JSON or JSONLINES formatted string.
        file_path (str): Path to file.
    """
    if (file_path.split('.') == '.jsonlines') & ('\n' not in data):
        raise ValueError('Data must contain new lines to be written to a .jsonlines file.')
    with open(file_path, 'w') as f:
        f.write(data)


class TFRecordExamples:
    pass
