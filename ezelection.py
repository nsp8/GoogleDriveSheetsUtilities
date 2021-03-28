import re
import pandas as pd
import util


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
        for x, row in zip(_values.index, _values.to_dict(orient='records')):
            _previous = row[col_name]
            _rectified = _handler(_previous, row[voter_column], _pattern)
            _df.loc[x, col_name] = _rectified
        return _df

    non_unique_cols = list()
    non_unique_values = list()
    for col in data_frame.columns:
        unique_rows = data_frame[col].nunique()
        if unique_rows != data_frame.shape[0]:
            non_unique_cols.append(col)
            values = pd.Series(data_frame[col].unique())
            unique_values = util.remove_empty_strings(values)
            non_unique_values.append(unique_values)
    df_copy = data_frame.copy(deep=True)
    write_in_col = non_unique_cols[-1]
    write_in_values = non_unique_values[-1]
    df_1 = clean_prefixes("self", df_copy, write_in_col, write_in_values)
    df_2 = clean_prefixes("n votes to", df_1, write_in_col, write_in_values)
    return df_2


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

    # data_blocks = util.extract_data_blocks(file_path)
    # internal_frames = util.split_internal_frames(data_blocks)
    # frames = [util.correct_headers(f)[0]
    #           for f in map(util.clean_data_frame, internal_frames.values())]
    # main_df = frames[-1].copy(deep=True)
    voter_column = "Voter"
    proxy_voter_column = main_df[voter_column].str.extract(r"\((.*)\)")
    main_df_copy = main_df.copy(deep=True)
    first_column = main_df[main_df.columns[0]]
    if is_first_col_index(first_column):
        main_df_copy.drop([main_df.columns[0]], axis=1, inplace=True)
    main_df_copy.set_index(voter_column, inplace=True)
    just_numbers = main_df_copy.applymap(text_remover)
    # noinspection PyTypeChecker
    total_votes = just_numbers.apply(sum, axis=1)
    """
    reindex = main_df[main_df[voter_column].sort_values() == 
    total_votes.sort_index().index].index
    total_column = total_votes.reset_index().set_index(reindex)
    """
    main_df["PROXY-VOTER"] = proxy_voter_column
    main_df["TOTAL"] = total_votes.to_list()
    return main_df


def get_data_local(file_path):
    """
    Reads a local data file and tries to splits it into constituent DataFrames.
    :param file_path: str - path of the file to read.
    :return: list of DataFrames.
    """
    data_blocks = util.extract_data_blocks(file_path)
    internal_frames = util.split_internal_frames(data_blocks)
    frames = [util.correct_headers(f)[0]
              for f in map(util.clean_data_frame, internal_frames.values())]
    return frames
