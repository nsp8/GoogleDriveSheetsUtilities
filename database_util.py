import pandas as pd
import google_drive as gd


def get_data_from_tic(df):
    """
    Parses Table Instance Charts (TIC) from the DataFrame and returns list
    of columns and data values.
    :param df: DataFrame corresponding to the TIC.
    :return: tuple - list of columns of the table, list of list of records.
    """
    target_row_id = "Sample Row 1"
    dfc = df.copy(deep=True)
    dfc.set_index(dfc.columns[0], inplace=True)
    i = dfc.index.to_list().index(target_row_id)
    sub = dfc[i:].to_dict(orient="records")
    col_list = sub[0].keys()
    values = list()
    for r in sub:
        values.append(
            [v for v in map(
                lambda s: f"'{s}'" if not str.isnumeric(s) else s,
                r.values()
            )]
        )
    return col_list, values


def dfs_to_insert_stmt(df_dict):
    """
    Parses the dict of DataFrames and returns Insert statements for each table.
    :param df_dict: dict - of DataFrames of TICs.
    :return: dict - key  :  table-name
                    value:  list of insert statements from each TIC DataFrame.
    """
    insert_statements = dict()
    for table_name, dataframe in df_dict.items():
        col_list, rows = get_data_from_tic(dataframe)
        stmt_list = list()
        stmt = f"INSERT INTO {table_name} ({','.join(col_list)}) VALUES"
        for row in rows:
            _string = ','.join(row)
            _insert = f"{stmt} ({_string});"
            stmt_list.append(_insert)
        insert_statements[table_name] = stmt_list
    return insert_statements


def get_table_instance_charts():
    """
    Returns all Table Instance Charts from Excel/Spreadsheet files with name
    like "table instance" (saved in Google Drive).
    :return: dict - key  :  file-name,
                    value:  dict (key: Table Name, value: DataFrame of TIC)
    """
    drive = gd.GoogleDriveAPI()
    file_subset = "table instance"
    sheets = drive.get_unique_spreadsheets_list()
    sheets_df = pd.DataFrame(sheets)
    _condition = sheets_df.name.str.lower().str.contains(file_subset)
    target_files = sheets_df[_condition]
    files = dict()
    for file in target_files.to_dict(orient="records"):
        _id, _name = file.get("id"), file.get("name")
        files[_name] = drive.sheet_to_df_dict(_id)
    return files
