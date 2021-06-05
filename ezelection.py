from os import path
import logging
import re
import pandas as pd
import util

logger = logging.getLogger("ezelection")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("ezelection.log")
logger.addHandler(file_handler)
formatter = logging.Formatter("%(asctime)s: "
                              "%(levelname)s: "
                              "%(name)s: "
                              "%(message)s")
file_handler.setFormatter(formatter)

VOTER_COLUMN_NAME = "Voter"
PROXY_VOTER_COLUMN_NAME = "PROXY-VOTER"
TOTAL_COLUMN_NAME = "TOTAL"
PROXY_VALUES_COLUMN = "GIVE PROXY VOTING DISCRETION"


def get_write_in_col(df):
    """
    Extracts the column name containing "Write-in" values.
    :param df - DataFrame that has values under a "Write-in" column.
    """
    df_cols = df.columns
    identifier = "[wW]rite"
    target_columns = df_cols[df_cols.str.contains(identifier)]
    if target_columns.shape[0] == 1:
        col_name = target_columns.item()
        return col_name
    # TODO: code for the case with multiple write-in columns:
    # elif target_columns.shape[0] > 1:
    #     return target_columns.to_list()
    return None


def clean_proxy_elections_block(data_frame):
    """
    Cleans the first block of data values (returned from ElectionBuddy)
    :param data_frame: Pandas DataFrame of the first block of data.
    :return: Cleansed/rectified DataFrame.
    """

    def correct_votes_to_prefixes(*args):
        """
        Helper function to correct the prefixes of the pattern:
        "all|n votes to proxy/voter" into "proxy/voter".
        :param args: list of arguments: string to replace, corresponding
        row-value at the voter column, pattern to match.
        :return: str - of the rectified value.
        """
        string, _, pattern = args
        match = re.search(pattern, string)
        if match:
            replacement = match.groups()[-1]
            return replacement.strip()
        return string

    def correct_self_strings(*args):
        """
        Helper function to correct the prefixes of the pattern:
        "Self(.*)" into "Self/Voter".
        :param args: list of arguments: string to replace, corresponding
        row-value at the voter column, pattern to match.
        :return: str - of the rectified value.
        """
        string, src_val, pattern = args
        replacement = re.subn(r"\(.*\)", "", src_val)[0].strip()
        new_string = re.subn(pattern, f"Self/{replacement}", string.lower())
        return new_string[0].strip()

    def clean_prefixes(prefix_type, df, col_name, col_values):
        """
        Initiates cleaning of a given `prefix_type` from the "Write-in"
        column of the DataFrame `df`.
        :param prefix_type: str - type of prefix to rectify.
        :param df: Pandas DataFrame.
        :param col_name: str - column name to target.
        :param col_values: list - of column values.
        :return: The rectified DataFrame.
        """
        _types = {
            # r"([\S+]+)\s*(votes\s*to\s*)(.*)"
            "n votes to": {
                "pattern": r"(.*)\s*(vote[s]?\s*to\s*)(.*)",
                "handler": correct_votes_to_prefixes
            },
            "self": {
                "pattern": r"^self.*",
                "handler": correct_self_strings
            },
        }
        voter_column = "Voter"
        _df = df.copy(deep=True)
        _pattern = _types[prefix_type].get("pattern")
        _handler = _types[prefix_type].get("handler")
        _match = col_values.str.lower().str.contains(_pattern, regex=True)
        _prefixed = col_values[_match]
        _values = _df[_df[col_name].isin(_prefixed)]
        for i, row in zip(_values.index, _values.to_dict(orient="records")):
            _previous = row[col_name]
            _rectified = _handler(_previous, row[voter_column], _pattern)
            _df.loc[i, col_name] = _rectified
        return _df

    df_copy = data_frame.copy(deep=True)
    write_in_col = get_write_in_col(df_copy)
    df_copy[write_in_col] = df_copy[write_in_col].apply(str.strip)
    write_in_values = util.remove_empty_strings(df_copy[write_in_col])
    df_1 = clean_prefixes("self", df_copy, write_in_col, write_in_values)
    df_2 = clean_prefixes("n votes to", df_1, write_in_col, write_in_values)
    df_3 = df_2.applymap(lambda s: re.subn(r'\"*', '', s.strip())[0])
    return df_3


def apply_aggregations_local(main_df):
    """
    Applies required aggregations on the Excel files for EZ Election Solutions
    :param main_df: DataFrame - on which aggregations will be applied.
    :return: DataFrame with the aggregations.
    """

    def text_remover(value):
        """
        Replaces with any textual instances with 0 (for summation of
        numerical data).
        :param value: cell value of the DataFrame
        :return: int of `value`, if `value` was purely numeric, otherwise 0.
        """
        data = str(value).strip()
        return int(data) if str.isnumeric(data) else 0

    def is_first_col_index(first_col):
        """
        Checks if the first column comprises of sequential index-values
        (for removal from the DataFrame).
        :param first_col: Pandas Series of the first column of the main DF.
        :return: bool - True if the condition is satisfied, False otherwise.
        """
        first_col_values = [int(x) for x in first_col.copy(deep=True)]
        first_col_values.sort()
        first_val = first_col_values[0]
        _stop = first_col.shape[0] + first_val
        _scaled = first_col_values == list(range(first_val, _stop))
        return _scaled

    voter_column = main_df[VOTER_COLUMN_NAME].str.extract(r"\((.*)\)")
    main_df_copy = main_df.copy(deep=True)
    first_column = main_df[main_df.columns[0]]
    if is_first_col_index(first_column):
        main_df_copy.drop([main_df.columns[0]], axis=1, inplace=True)
    main_df_copy.set_index(VOTER_COLUMN_NAME, inplace=True)
    just_numbers = main_df_copy.applymap(text_remover)
    # noinspection PyTypeChecker
    total_votes = just_numbers.apply(sum, axis=1)
    main_df[PROXY_VOTER_COLUMN_NAME] = voter_column
    main_df[TOTAL_COLUMN_NAME] = total_votes.to_list()
    return main_df


def get_frames(data_frame):
    """
    Splits the input DataFrame into constituent DataFrames.
    :param data_frame: DataFrame to split into parts
    :return: list of constituent DataFrames.
    """
    try:
        internal_frames = util.split_internal_frames(data_frame)
        frames = [util.correct_headers(f)[0]
                  for f in map(util.clean_data_frame,
                               internal_frames.values())]
        return frames
    except Exception as e:
        logger.error(f"Encountered an error in get_frames: {e}")
        return list()


def get_data_local(file_path):
    """
    Reads a local data file and tries to returns the constituent DataFrames.
    :param file_path: str - path of the file to read.
    :return: list of DataFrames.
    """
    # return get_frames(util.extract_data_blocks(file_path))
    return util.extract_data_blocks(file_path)


def get_data_from_drive(folder_name, file_name):
    """
    Returns DataFrames from a file from a folder in Google Drive
    :param folder_name: name or subset of name of the parent folder
    :param file_name: name or subset of name of the file to fetch
    :return: list of DataFrames.
    """
    try:
        import google_drive as drive
        drive_api = drive.GoogleDriveAPI()
        proxy_vote_result_folders = drive_api.get_folder(folder_name, True)
        if proxy_vote_result_folders:
            if "children" in proxy_vote_result_folders:
                folder_contents = proxy_vote_result_folders.get("children")
                file = None
                if file_name:
                    for _object in folder_contents:
                        if _object:
                            if file_name in _object.get("name"):
                                file = _object
                                break
                    if file:
                        file_type = drive_api.get_file_type_from_mime(
                            file.get("mimeType"))
                        file_data = util.get_file_data(file, file_type)
                        file_df = file_data[list(file_data.keys())[0]]
                        return get_frames(file_df)
                    logger.error(f"Could not find file '{file_name}' "
                                 f"in folder '{folder_name}'")
    except Exception as e:
        logger.error(f"Encountered an error in get_data_from_drive: {e}")
    return None


def equalize_columns(*data_frames):
    """
    Equalize the column datatype of the the DataFrames to str.
    :param data_frames: DataFrames passed as arguments
    :return: list of DataFrames
    """
    df_list = list()
    for data_frame in data_frames:
        df = data_frame.copy(deep=True)
        for col in df.columns:
            df[col] = df[col].astype("str")
        df_list.append(df)
    return df_list


def get_org_heads(data_frame, pattern):
    """
    Parses the DataFrame to get the list of organization heads.
    :param data_frame: DataFrame to parse.
    :param pattern: regex pattern to use to search.
    :return: modified DataFrame with cleansed column head and
    list of org-heads found.
    """
    processed_cols = list()
    df = data_frame.copy(deep=True)
    df_cols = df.columns
    column_list = df_cols[df_cols.str.contains(util.REPLACEMENT,
                                               regex=False)].to_list()
    for col in column_list:
        _spl = col.split(util.REPLACEMENT)
        if processed_cols:
            last_added = processed_cols[-1]
            if last_added in _spl:
                logger.info("Value already added.")
            else:
                processed_cols.append(_spl[-1])
        else:
            _match = re.search(pattern, _spl[0])
            if _match:
                _group = _match.groups()[0]
                processed_cols.append(_group)
        new_col = {col: re.subn(util.REPLACEMENT, ",", col)[0]}
        df.rename(columns=new_col, inplace=True)
    return df, processed_cols


def apply_final_aggregations(first_df, second_df):
    """
    Applies aggregations to create the last DataFrame.
    :param first_df: the proxy voters DataFrame
    :param second_df: the vote-distribution DataFrame
    :return: the final DataFrame.
    """
    def get_row_total(x, y):
        """
        Folding/reducing function to sum values in a row (skipping strings).
        """
        try:
            if isinstance(x, str):
                if isinstance(y, str):
                    return 0
                return y
            elif isinstance(y, str):
                return x
            return x + y
        except Exception as err:
            logger.error(f"Encountered an error in get_row_total: {err}")

    from functools import reduce
    try:
        df1 = first_df.copy(deep=True)
        write_in_col = get_write_in_col(df1)
        write_in_values = util.remove_empty_strings(df1[write_in_col])
        write_in_condition = df1[write_in_col].isin(write_in_values)
        org_head_votes = df1[~write_in_condition]
        head_pattern = r"Option\s*:\s*(\w+(\W+\w+)+)"
        df1, org_heads = get_org_heads(df1, head_pattern)
        proxy_holders = list()
        proxy_holders.extend(org_heads)
        write_in_list = write_in_values.unique().tolist()
        write_in_list.sort()
        proxy_holders.extend(write_in_list)
        final_df_cols = ["PROXY HOLDER"]
        col_index = list(second_df.columns).index(PROXY_VALUES_COLUMN)
        col_list = second_df.columns[col_index:]
        final_df_cols.extend(col_list)
        final_df_cols.append("Total")
        df2 = second_df.copy(deep=True)
        df2 = df2.applymap(lambda s: re.subn(r'\"*', '', s.strip())[0])
        vote_dist_df = df2.applymap(util.convert_to_numeric)
        final_data = list()
        for i, p in enumerate(proxy_holders):
            data = {final_df_cols[0]: p}
            if i == 0:
                lookup_df_2 = vote_dist_df[VOTER_COLUMN_NAME].isin(
                    org_head_votes[VOTER_COLUMN_NAME])
            else:
                lookup_df_1 = df1[write_in_col].str.contains(p, regex=False)
                lookup_df_2 = vote_dist_df[VOTER_COLUMN_NAME].isin(
                    df1[lookup_df_1][VOTER_COLUMN_NAME])
            # Applying sum across a column: reducing string values to 0
            col_values = vote_dist_df[lookup_df_2][col_list].apply(
                lambda s: reduce(get_row_total, s.to_list())).to_dict()
            total = reduce(get_row_total, col_values.values())
            data.update(col_values)
            data.update({"Total": total})
            final_data.append(data)
        return pd.DataFrame(final_data)
    except Exception as e:
        logger.error(f"Encountered an error in apply_final_aggregations: {e}")
        return pd.DataFrame()


def test_file(local_file_path):
    """
    Tests new files and creates the third DataFrame for the data.
    :param local_file_path: str - file path.
    :return: DataFrame containing the processed data.
    """
    try:
        if path.exists(local_file_path):
            local_data = util.extract_data_blocks(local_file_path)
            assert len(local_data) > 1
            vote_distribution_df = local_data[1]
            clean_proxy_df = clean_proxy_elections_block(local_data[0])
            final_df = apply_final_aggregations(clean_proxy_df,
                                                vote_distribution_df)
            return final_df
    except AssertionError:
        logger.error("Could only find one frame in local file.")
    except Exception as e:
        logger.error(f"Error in test_file: {e}")
    return pd.DataFrame()


def test_drive_sample(local_file_map: dict, drive_file_map: dict):
    """
    Tests sample file data.
    :param local_file_map: dict of folder and file names/paths of local file.
    :param drive_file_map: dict of folder and file names/paths of Google Drive
    file.
    :return: a tuple of:
        Comparison DF - differences between the local and Drive files,
        Local file data-frame and Google Drive file data-frame.
    """
    required_keys = {"folder", "file"}
    try:
        _keys = set(local_file_map.keys()).intersection(
            set(drive_file_map.keys()))
        assert required_keys.issubset(_keys)
        local_folder = local_file_map.get("folder")
        local_file = local_file_map.get("file")
        file_path = path.join(local_folder, local_file)
        if path.exists(file_path):
            local_data = get_data_local(file_path)
            local_frame_index = 1 if len(local_data) > 0 else 0
            if local_frame_index == 0:
                raise Exception("Could only find one frame in local file.")
            vote_distribution_df = local_data[local_frame_index]
            applied_agg_df = apply_aggregations_local(vote_distribution_df)
            agg_sorted_df = applied_agg_df.sort_values(
                [TOTAL_COLUMN_NAME, PROXY_VOTER_COLUMN_NAME],
                ignore_index=True)
            drive_folder = drive_file_map.get("folder")
            drive_file = drive_file_map.get("file")
            drive_file_data = get_data_from_drive(drive_folder, drive_file)
            clean_proxy_df = clean_proxy_elections_block(local_data[0])
            final_df = apply_final_aggregations(clean_proxy_df,
                                                vote_distribution_df)
            drive_frame_index = 1 if len(drive_file_data) > 0 else 0
            drive_file_df = drive_file_data[drive_frame_index]
            last_row = drive_file_df.index[-1]
            last_value = drive_file_df.loc[last_row][VOTER_COLUMN_NAME]
            drive_agg_df = drive_file_df.copy(deep=True)
            if last_value.lower() in ["total"]:
                drive_agg_df = drive_agg_df.drop(index=last_row)
            drive_df_sorted = drive_agg_df.sort_values(
                [TOTAL_COLUMN_NAME, PROXY_VOTER_COLUMN_NAME],
                ignore_index=True).reset_index()
            is_subset = agg_sorted_df[PROXY_VOTER_COLUMN_NAME].isin(
                    drive_agg_df[PROXY_VOTER_COLUMN_NAME])
            local_df = agg_sorted_df[is_subset]
            local_df = local_df[drive_agg_df.columns]
            reqd_index = list(drive_df_sorted.columns).index(VOTER_COLUMN_NAME)
            reqd_cols = drive_df_sorted.columns[reqd_index:]
            local_df, drive_df_sorted = equalize_columns(local_df,
                                                         drive_df_sorted)
            eq_condition = local_df[reqd_cols].eq(drive_df_sorted[reqd_cols])
            comparison_df = local_df[~eq_condition][reqd_cols]
            comparison_df.dropna(axis=1, how="all", inplace=True)
            comparison_df.dropna(axis=0, how="all", inplace=True)
            return comparison_df, final_df
    except AssertionError as ae:
        logger.error(f"A required key wasn't in the dict: {ae}")
    except Exception as e:
        logger.error(f"Encountered an error in test_sample: {e}")
    return tuple()
