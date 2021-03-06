from os import path
import logging
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


def format_csv_data(file_path=None):
    if not file_path:
        src_path = r"fallback_folder_path"
        file_name = "fallback_file.csv"
        file_path = path.join(src_path, file_name)
    with open(file_path) as f:
        contents = f.readlines()

    def get_first_row_pos():
        content_map = [{"row": row, "len": len(row.split(","))} for row in
                       contents]
        meta_df = pd.DataFrame(content_map)
        most_occurrences = meta_df["len"].mode().squeeze()
        return meta_df[meta_df["len"] == most_occurrences].iloc[0].name

    data_df = pd.read_csv(file_path, skiprows=get_first_row_pos())
    return data_df


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


def correct_header(data_frame):
    _cols = data_frame.columns
    _if_blanks = _cols.str.contains("Unnamed")
    if _if_blanks.any():
        _next = data_frame.iloc[0]
        logger.info(f"next row: {_next.to_dict()}")
        _replacer = _next[_if_blanks]
        logger.info(f"replacement: {_replacer.to_dict()}")
        _start = _next.name
        logger.info(f"start index: {_start}")
        _blanks = _cols[_if_blanks].to_list()
        logger.info(f"blanks: {_blanks}")
        _not_blanks = _cols[~_if_blanks].to_list()
        logger.info(f"not_blanks: {_not_blanks}")
        _not_blanks.remove("count")
        _combiner = _next[~_if_blanks]
        _head = {old: new for old, new in zip(_blanks, _replacer)}
        _updates = {old: f"{old}_{new}" for old, new in zip(_not_blanks,
                                                            _combiner)}
        _head.update(_updates)
        logger.info(f"new-map: {_head}")
        df = data_frame.copy(deep=True)
        df.rename(columns=_head, inplace=True)
        df.drop([_start], inplace=True)
        return df, True
    return data_frame, False


def split_internal_frames(data_frame):
    frames = dict()
    prev_df = data_frame.copy(deep=True)
    prev_df["count"] = prev_df.apply(pd.Series.count, axis=1)
    df_breaks = prev_df[prev_df["count"] == 0] + prev_df[prev_df["count"] == 1]
    for i in range(len(df_breaks.index)):
        _start = df_breaks.index[i] + 1
        try:
            _end = df_breaks.index[i + 1]
        except IndexError:
            _end = -1
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
