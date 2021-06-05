# from os import path
from copy import deepcopy
import logging
import re
import pandas as pd

logger = logging.getLogger("util")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("util.log")
logger.addHandler(file_handler)
formatter = logging.Formatter("%(asctime)s: "
                              "%(levelname)s: "
                              "%(name)s: "
                              "%(message)s")
file_handler.setFormatter(formatter)


REPLACEMENT = "#*#"


def df_to_list(data_df: pd.DataFrame):
    """
    Converts a DataFrame to a list.
    :param data_df: DataFrame to be converted.
    :return: list.
    """
    data_parts = data_df.to_dict(orient="split")
    cols_ = data_parts["columns"]
    data = data_parts["data"]
    data.insert(0, cols_)
    return data


def list_to_df(data_list):
    """
    Converts a list of data rows into a DataFrame.
    :param data_list: list of values
    :return: Pandas DataFrame of the values in `data_list`
    """
    def split_rows(row):
        separator = "*#*"
        return [re.sub(r"\n|\"+", "", r.replace(separator, ", ")) for r in
                row.replace(", ", separator).split(",")]
    _rows = [r for r in map(split_rows, data_list)]
    return pd.DataFrame(_rows[1:], columns=_rows[0])


def get_first_row_pos(contents: list):
    """
    Determines the starting position/index of data values.
    :param contents: list - of string or list values.
    :return: int - index position where the data values start from.
    """
    try:
        content_map = [{"row": row,
                        "len":
                            len(row.split(",")) if isinstance(row, str) else
                            len(row) if isinstance(row, list) else 0}
                       for row in contents]
        meta_df = pd.DataFrame(content_map)
        _mode = meta_df["len"].mode()
        if _mode.shape[0] > 1:
            _mode = _mode.max()
        most_occurrences = _mode.squeeze()
        return meta_df[meta_df["len"] == most_occurrences].iloc[0].name
    except Exception as e:
        logger.info("Exception occurred in <get_first_row_pos>")
        logger.error(e)
        return 0


def get_frame_indices(meta_frame):
    """
    Calculates the beginning and ending indices of data blocks.
    :param meta_frame: DataFrame containing metadata on the rows
    (rows with type of spacing around commas in strings and split-length)
    :return: list of dictionaries of start and end indices.
    """
    indices = list()
    len_series = meta_frame["len"]
    blank_df = meta_frame[len_series == len_series.min()]
    data_blocks = meta_frame[len_series > len_series.min()]
    for blank_row in blank_df.index:
        _data = dict()
        current_row_len = meta_frame.loc[blank_row]["len"]
        _next = blank_row + 1
        if _next in meta_frame.index:
            next_row = meta_frame.loc[_next]
            next_row_len = next_row["len"]
            if next_row_len > current_row_len:
                if len(indices):
                    data_index = data_blocks.index
                    i = data_index.to_list().index(_next)
                    indices[-1].setdefault("end", data_index[i-1])
                _data["start"] = _next
                indices.append(_data)
    indices[-1].setdefault("end", data_blocks.index[-1])
    return indices


def dict_to_df(sheet_data: dict):
    """
    Formats a dict of sheet values into a pandas DataFrame.
    :param sheet_data: dict - containing the values to convert to a DataFrame.
        The dict must have a key called "values" that holds all the data.
        The first "row" will be used to form the header of the DF.
    :return: pandas.DataFrame - of the values, if everything goes right,
        otherwise an empty DataFrame.
    """
    try:
        sheet_values = sheet_data.get("values", None)
        if sheet_values:
            logger.info(f"sheet_values = {sheet_values}")
            i = get_first_row_pos(sheet_values)
            sheet_header, sheet_data = sheet_values[i], sheet_values[i+1:]
            sheet_df = pd.DataFrame(data=sheet_data,
                                    columns=sheet_header,
                                    index=None)
            logger.info(sheet_df)
            return sheet_df
    except Exception as e:
        logger.info("Encountered an exception <dict_to_df>")
        logger.error(e)
        return pd.DataFrame()


def get_spreadsheets_from_files(id_list, mime_types, file_name="inputs"):
    """
    Fetches a list of all Google Sheets file IDs that have the given
    `file_name`.
    :param id_list: list of dicts of file IDs to search through.
    :param mime_types: list - of MIME types to check.
    :param file_name: str - name of the file to search.
    :return: list - of Google Sheets files.
    """
    sheets = mime_types["spreadsheet"]
    spreadsheets = list()
    for file_id in id_list:
        if file_id["mimeType"] == sheets and file_id["name"] == file_name:
            spreadsheets.append(file_id)
    return spreadsheets


def perform_split(contents):
    """
    Perform string split on comma separated values without breaking commas in
    string values.
    :param contents: list of file rows
    :return: list of cleaner file rows with appropriate parsing
    """
    from numpy import nan
    new_contents = deepcopy(contents)
    row_metadata = get_row_metadata(new_contents)
    meta_df = pd.DataFrame(row_metadata).T
    meta_df_reduced = meta_df.applymap(lambda s: nan if not s else s).dropna(
        axis=1, how="all")
    valid_df = meta_df[meta_df["col_width"] > 1]
    value_counts = valid_df["col_width"].value_counts().reset_index()
    counts = value_counts["col_width"]
    outliers = value_counts[counts == counts.min()]["index"]
    # dirty_rows = meta_df_reduced[meta_df_reduced["col_width"] == outlier]
    dirty_rows = meta_df_reduced[meta_df_reduced["col_width"].isin(outliers)]
    dirty_rows = dirty_rows.dropna(axis=1, how="all")
    column = list(set(dirty_rows.columns) - {"row", "col_width"})[0]
    patterns = dirty_rows[column].unique()[0]
    pattern = ""
    if isinstance(patterns, tuple):
        pattern = patterns[0]
    dirty_rows["row"] = dirty_rows["row"].apply(
        lambda s: re.subn(pattern, REPLACEMENT, s)[0])
    for i in dirty_rows.index:
        new_contents[i] = dirty_rows["row"].loc[i]
    return new_contents


def extract_data_blocks(file_path):
    """
    Extracts the largest block of data in a CSV file that has many.
    :param file_path: str - local path of the CSV file
    :return: DataFrames of each block of data if the parsing went
    well, otherwise an empty DataFrame.
    """

    def create_content_map(_contents):
        """
        Parses the contents of the file (as rows) and creates a "meta" schema.
        :param _contents: list of rows.
        :return: list of dictionaries of metadata on the rows.
        """
        _content_map = list()
        for row in _contents:
            _map = dict()
            _row = re.subn(r"^(\W|_)+|^(ï»¿\w)+", "", row)[0]
            _len = 0
            if isinstance(_row, str):
                _len = len(_row.split(","))
            elif isinstance(_row, list):
                _len = len(_row)
            _map["row"] = _row
            _map["len"] = _len
            _content_map.append(_map)
        return _content_map

    dataframes = list()
    try:
        with open(file_path) as f:
            contents = f.readlines()

        new_content = perform_split(contents)
        content_map = pd.DataFrame(create_content_map(new_content))
        frame_indices = get_frame_indices(content_map)
        for frame in frame_indices:
            start, end = frame["start"], frame["end"]
            rows_str = new_content[start:end + 1]
            rows_list = [re.subn(r"\n", "", r)[0].split(",") for r in rows_str]
            df = pd.DataFrame(data=rows_list[1:], columns=rows_list[0])
            dataframes.append(df)
    except Exception as e:
        logger.error(f"Couldn't format CSV data because: {e}")
    return dataframes


def clean_data_frame(data_frame: pd.DataFrame):
    """
    Performs basic clean-up of a DataFrame.
    :param data_frame: input DataFrame.
    :return: a cleaner DataFrame.
    """
    data_frame.dropna(axis=0, inplace=True, how="all")
    data_frame.dropna(axis=1, inplace=True, how="all")
    data_frame.fillna('', axis=0, inplace=True)
    return data_frame


def correct_headers(data_frame):
    """
    Rectifies the DataFrame's header if it has Unnamed columns
    (shifted/translated frame values)
    :param data_frame: Pandas DataFrame of values
    :return: tuple - DataFrame, boolean: rectified DF, True if there were
    unnamed values in the header otherwise the original DF, False.
    """
    data_frame = clean_data_frame(data_frame.copy(deep=True))
    _cols = data_frame.columns
    _unnamed = _cols.str.contains("Unnamed")
    # | _cols.str.contains(r"^$",regex=True)
    if _unnamed.any():
        _next = data_frame.iloc[0]
        logger.info(f"next row: {_next.to_dict()}")
        _replacer = _next[_unnamed]
        logger.info(f"replacement: {_replacer.to_dict()}")
        _start = _next.name
        logger.info(f"start index: {_start}")
        _blanks = _cols[_unnamed].to_list()
        logger.info(f"blanks: {_blanks}")
        _not_blanks = _cols[~_unnamed].to_list()
        logger.info(f"not_blanks: {_not_blanks}")
        if "count" in _not_blanks:
            _not_blanks.remove("count")
        _combiner = _next[~_unnamed]
        _head = {old: new for old, new in zip(_blanks, _replacer)}
        _updates = {old: f"{old}_{new}" for old, new in zip(_not_blanks,
                                                            _combiner)}
        _head.update(_updates)
        logger.info(f"new-map: {_head}")
        df = data_frame.copy(deep=True)
        df.rename(columns=_head, inplace=True)
        df.drop([_start], inplace=True)
        if "count" in df.columns:
            df.drop("count", axis=1, inplace=True)
        return correct_headers(df)
    return data_frame, False


def split_internal_frames(data_frame):
    """
    Splits the DataFrame into its internal constituents (if present).
    :param data_frame: Pandas DataFrame
    :return: dict - if internal blocks found - key: (start, end) tuple of
    indices; value: block of DF, otherwise an empty dict.
    """
    frames = dict()
    prev_df = data_frame.copy(deep=True)
    prev_df["count"] = prev_df.apply(pd.Series.count, axis=1)
    df_breaks = prev_df[prev_df["count"].isin([0, 1])]
    for i in range(len(df_breaks.index)):
        _start = df_breaks.index[i] + 1
        try:
            _end = df_breaks.index[i + 1]
            if _start > _end:
                _end = None
        except IndexError:
            _end = None
        _df = prev_df[_start:_end]
        if _df.shape[0] < 2:
            logger.info("Empty")
        else:
            logger.info(_df.shape)
            _df.dropna(axis=1, how="all", inplace=True)
            frames[(_start, _end)] = _df
    return frames


def get_file_data(file_object, file_type="csv"):
    """
    Converts CSV/Excel file (basis `file_type`) into a DataFrame (CSV),
    or a dict where the keys are sheet names and the corresponding values
    are DataFrames.
    :param file_object: dict - containing the ID of the file.
    :param file_type: str - should be either "csv", "xls[x]" or "excel".
    :return: pandas DataFrame (for csv) or dict (excel); None in case of
    errors or if not data was retrieved.
    """
    assert file_type is not None
    _type = file_type.strip().lower()
    assert _type in ["csv", "excel", "xls", "xlsx"]

    from requests import get
    from io import StringIO
    from pandas import read_csv, read_excel

    try:
        file_id = file_object["id"]
        _url = f"https://drive.google.com/uc?export=download&id={file_id}"
        _response = get(_url)
        if _response:
            if _type == "csv":
                _content = _response.text
                _raw = StringIO(_content)
                return read_csv(_raw)
            elif _type in ["excel", "xls", "xlsx"]:
                _content = _response.content
                return read_excel(_content, sheet_name=None)
        else:
            logger.info(f"Response was not found for: {_url}")
            return None
    except AssertionError as assertion:
        logger.error(f"<get_file_data>: {assertion}\n"
                     f"file_type should either be 'csv' or 'excel'.")
        return None
    except KeyError as err:
        logger.error(
            f"<get_file_data>: {err} not found in {file_object.keys()}")
        return None
    except Exception as e:
        file_name = file_object.get("name")
        logger.error(
            f"<get_file_data>: Caught Exception \t[File: {file_name}] {e}")
        return None


def get_internal_frame_indices(contents: list):
    """
    Breaks the `contents` (list) on the basis of empty strings and returns the
    indices of these breakpoints.
    :param contents: list of textual data
    :return: list of indices
    """
    data_rows = deepcopy(contents)
    _splits = list()
    _count = 0
    for _index, _row in enumerate(data_rows):
        i = _index + _count
        v = _row.strip()
        if not v and i not in _splits:
            _splits.append(i)
            data_rows.pop(_index)
            _count += 1
    return _splits


def get_internal_frames(contents):
    """
    Returns a list of DataFrames of the different sections of tabular data
    present in `contents`.
    :param contents: list of textual data
    :return: list of DataFrames
    """
    _indices = get_internal_frame_indices(contents)
    df_list = list()
    for i in range(len(_indices) - 1):
        pos, nxt = _indices[i]+1, _indices[i+1]
        _slice = contents[pos:nxt]
        _df = list_to_df(_slice)
        if _df.shape[0] > 10:
            df_list.append(_df)
    return df_list


def get_row_metadata(rows):
    """
    Returns metadata about the CSV rows' multiply-delimited values.
    Use this for debugging against the contents of the CSV.
    :param rows: list of values
    :return: dict of metadata
    """
    metadata = dict()
    space_pattern_map = {
        "delimiter": r"(\w+),\s+(\w+)\s+,(\w+)",
        "multiple": r"(\s+,\s+)",
        "before": r"(\s+,\s?)",
        "after": r"(\s?,\s+)"
    }
    try:
        for i, row in enumerate(rows):
            if isinstance(row, list):
                row = ",".join(str(x) for x in row)
            metadata[i] = dict()
            metadata[i]["row"] = row
            _replacer = re.subn(r"\s+,\s+", "*#*", row)
            _values = _replacer[0].split(",")
            for _type, _pattern in space_pattern_map.items():
                _match = re.search(_pattern, row)
                if _match:
                    _groups = _match.groups()
                    metadata[i][_type] = _groups
                else:
                    metadata[i][_type] = tuple()
            metadata[i]["col_width"] = len(_values)
    except Exception as e:
        logger.info(f"Exception in get_row_metadata: {e}")
    return metadata


def remove_empty_strings(data_series):
    """
    Removes the Series of data values that do not contain empty strings.
    :param data_series: Pandas Series
    :return: Series with removed empty string data.
    """
    # _series = data_series.apply(str.strip)
    return data_series[~data_series.str.contains(r"^$", regex=True)]


# noinspection PyTypeChecker
def series_difference(subtrahend: pd.Series, minuend: pd.Series):
    """
    Returns the set difference of two Series objects.
    :param subtrahend: Pandas Series - series of data being subtracted.
    :param minuend: Pandas Series - series of data to subtract from the first.
    :return: set - of the difference between two Series.
    """
    _minuend = set(minuend.apply(str.strip))
    _subtrahend = set(subtrahend.apply(str.strip))
    return _subtrahend - _minuend


def convert_to_numeric(value):
    """
    Converts a cell-value of a DataFrame to its numeric value if it is valid.
    :param value: cell-value, presumably a string
    :return: numeric equivalent of `value` otherwise the value itself.
    """
    try:
        if isinstance(value, str):
            if re.search(r"(under|over)vote[d]", value.lower()):
                return 0
            if value.strip() == '':
                return 0
            if value.isdigit():
                return int(value)
            else:
                n = float(value)
                return n
    except ValueError as v:
        logger.error(f"Encountered ValueError in convert_to_numeric: {v}")
    except Exception as e:
        logger.error(f"Encountered an exception in convert_to_numeric: {e}")
    return value
