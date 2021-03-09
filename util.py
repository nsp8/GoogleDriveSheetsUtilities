# from os import path
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


def dict_to_df(sheet_data: dict):
    """
    Formats a dict of sheet values into a pandas DataFrame.
    :param sheet_data: dict - containing the values to convert to a DataFrame.
        The dict must have a key called "values" that holds all the data.
        The first "row" will be used to form the header of the DF.
    :return: pandas.DataFrame - of the values, if everything goes right,
        otherwise an empty DataFrame.
    """
    sheet_values = sheet_data.get("values", None)
    if sheet_values:
        logger.info(f"sheet_values = {sheet_values}")
        sheet_header, sheet_data = sheet_values[0], sheet_values[1:]
        sheet_df = pd.DataFrame(data=sheet_data,
                                columns=sheet_header,
                                index=None)
        return sheet_df
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


def extract_data_blocks(file_path):
    """
    Extracts the largest block of data in a CSV file that has many.
    :param file_path: str - local path of the CSV file
    :return: DataFrame of the largest block of data if the parsing went 
    well, otherwise an empty DataFrame.
    """
    try:
        with open(file_path) as f:
            contents = f.readlines()

        def get_first_row_pos():
            content_map = [{"row": row, "len": len(row.split(","))} for row in
                           contents]
            meta_df = pd.DataFrame(content_map)
            _mode = meta_df["len"].mode()
            if _mode.shape[0] > 1:
                _mode = _mode.max()
            most_occurrences = _mode.squeeze()
            return meta_df[meta_df["len"] == most_occurrences].iloc[0].name

        data_df = pd.read_csv(file_path, skiprows=get_first_row_pos())
        return data_df
    except Exception as e:
        logger.error(f"Couldn't format_csv_data because: {e}")
        return pd.DataFrame()


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
    import requests
    from io import StringIO
    try:
        file_id = file_object["id"]
        _url = f"https://drive.google.com/uc?export=download&id={file_id}"
        _response = requests.get(_url)
        if _response:
            if file_type.strip().lower() == "csv":
                _content = _response.text
                _raw = StringIO(_content)
                return pd.read_csv(_raw)
            elif file_type.strip().lower() in ["excel", "xls", "xlsx"]:
                _content = _response.content
                return pd.read_excel(_content, sheet_name=None)
        else:
            logger.info(f"Response was not found for: {_url}")
            return None
    except AssertionError as assertion:
        logger.error(f"<get_file_data>: {assertion}\n"
                     f"file_type should either be 'csv' or 'excel'.")
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
    from copy import deepcopy
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
    for i, row in enumerate(rows):
        if isinstance(row, list):
            row = ",".join(str(x) for x in row)
        delimiter_matches = re.search(r"(\w+),\s+(\w+)\s+,(\w+)", row)
        multiple_spaces = re.search(r"(\s+,\s+)", row)
        spaces_before = re.search(r"(\s+,\s?)", row)
        spaces_after = re.search(r"(\s?,\s+)", row)
        metadata[i] = dict()
        metadata[i]["row"] = row
        metadata[i]["delimiter_matches"] = tuple()
        metadata[i]["multiply_spaced_matches"] = tuple()
        metadata[i]["spaces_before"] = tuple()
        metadata[i]["spaces_after"] = tuple()
        _replacer = re.subn(r"\s+,\s+", "*#*", row)
        _values = _replacer[0].split(",")
        metadata[i]["col_width"] = len(_values)
        if delimiter_matches:
            metadata[i]["delimiter_matches"] = delimiter_matches.groups()
        if multiple_spaces:
            metadata[i]["multiply_spaced_matches"] = multiple_spaces.groups()
        if spaces_before:
            metadata[i]["spaces_before"] = spaces_before.groups()
        if spaces_after:
            metadata[i]["spaces_after"] = spaces_after.groups()
    return metadata
