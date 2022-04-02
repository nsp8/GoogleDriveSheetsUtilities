from io import BytesIO
from os import path
from re import search
from string import ascii_uppercase as alphabets
import logging
from apiclient import errors
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import util

logger = logging.getLogger("google_drive_api")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("google_drive_api.log")
logger.addHandler(file_handler)
formatter = logging.Formatter("%(asctime)s: "
                              "%(levelname)s: "
                              "%(name)s: "
                              "%(message)s")
file_handler.setFormatter(formatter)

APP_SCOPES = ["https://www.googleapis.com/auth/drive"]


def authorize_user() -> Credentials:
    # Instantiate Google Drive and Spreadsheets service objects.
    # [Reads the tokens and credentials files for authentication.]
    creds = None
    token_file_exists = path.exists("token.json")
    if token_file_exists:
        creds = Credentials.from_authorized_user_file("token.json")

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        logger.info("Invalid credentials or token. Retrying...")
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Token expired! Refreshing...")
                creds.refresh(Request())
            except RefreshError as e:
                logger.error(f"Couldn't refresh credentials because:\n{e}.")
                logger.error("Retrying...")
                if token_file_exists:
                    r = util.move_file("token.json")
                    logger.info(f"token file moved to temp folder: {r}")
                    authorize_user()
        else:
            _file = 'credentials.json'
            flow = InstalledAppFlow.from_client_secrets_file(_file, APP_SCOPES)
            creds = flow.run_local_server(port=0)
            if creds.valid:
                logger.info("User authorization successful!")
            else:
                logger.error(f"User authorization failed!")
                raise RefreshError("User authorization failed!")
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds


class GoogleDriveAPI(object):
    def __init__(self):
        self.drive_service = None
        self.sheets_service = None
        self.google_mime_types = {
            "spreadsheet": "application/vnd.google-apps.spreadsheet",
            "folder": "application/vnd.google-apps.folder"
        }
        self.excel_mime_types = {
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet"
        }
        self.file_fields = ["kind", "id", "name", "mimeType", "trashed",
                            "parents"]
        try:
            creds = authorize_user()
            self.drive_service = build('drive', 'v3', credentials=creds,
                                       cache_discovery=False)
            self.sheets_service = build('sheets', 'v4', credentials=creds,
                                        cache_discovery=False)
        except RefreshError as e:
            print(f"Could not create the API services because: {e}")

    def get_file_metadata_by_query(self, query: str) -> list:
        """
        Returns a list of responses based on the query.
        :param query: str - query string to query the Drive API.
        :return: list of responses.
        """
        assert self.drive_service is not None
        result = list()
        try:
            _fields = f"files({', '.join(self.file_fields)})"
            response = self.drive_service.files().list(
                q=query, fields=_fields).execute()
            result = response.get("files", [])
        except errors.HttpError as error:
            logger.error(f"ERROR in <get_file_metadata_by_query>: {error}")
        except Exception as e:
            logger.error(f"ERROR in <get_file_metadata_by_query>: {e}")
        finally:
            return result

    def get_file_metadata_by_id(self, file_id: str):
        """
        Returns a list of responses based on the query.
        :param file_id: str - ID of the file.
        :return: list of responses.
        """
        assert self.drive_service is not None
        response = dict()
        try:
            _fields = ', '.join(self.file_fields)
            response = self.drive_service.files().get(
                fileId=file_id, fields=_fields).execute()
        except errors.HttpError as error:
            logger.error(f"ERROR in <get_file_metadata_by_id>: {error}")
        except Exception as e:
            logger.error(f"ERROR in <get_file_metadata_by_id>: {e}")
        finally:
            return response

    def get_folder_contents(self,
                            folder_id: str,
                            only_sub_folders: bool = False):
        """
        Retrieves a list of dicts of files/folders associated with the
        folder_id.
        :param folder_id: str - ID of the folder.
        :param only_sub_folders: bool - if only folders are to be retrieved.
        :return: list of dicts if contents were found, otherwise None.
        """
        folders_mime = f"mimeType='{self.google_mime_types['folder']}'"
        if only_sub_folders:
            query = f"{folders_mime} and '{folder_id}' in parents"
        else:
            query = f"'{folder_id}' in parents"
        try:
            all_files = self.get_file_metadata_by_query(query)
            if len(all_files) == 0:
                return None
            else:
                return all_files
        except errors.HttpError as error:
            logger.error(f"ERROR in get_folder_contents: {error}")
            return None

    def get_folder_id(self, results: list, check_parents: bool = True):
        """
        Retrieves the folder ID on the basis of an execute response
        {result of execute(): eg = drive_service.files().list(...).execute()}.
        :param results: list - of results of an execute() on a service object.
        :param check_parents: bool - searches within parents if True.
        :return: str - of folder ID if only one present, otherwise a list of
        IDs.
        """
        if not results:
            logger.info("No files found.")
        else:
            logger.info("Found target")
            if len(results) == 1:
                logger.info(f"\tFound folder: {results[0]['name']}")
                return results[0]['id']
            else:
                logger.info("Many folders")
                file_ids = list()
                for folder in results:
                    logger.info(f"\tTrying folder: {folder['name']}")
                    folder_id = folder['id']
                    logger.info(f"\t\tFile ID: {folder_id}")
                    if check_parents:
                        _file_id = self.get_folder_contents(folder_id)
                        if _file_id:
                            return folder_id
                    else:
                        file_ids.append(folder_id)
                if file_ids:
                    return file_ids
                return None

    def get_folder(self, folder_name: str, is_name_subset: bool = False):
        """
        Fetches a dict of a given folder's ID and the children contained inside
        (if present), even the ones in "Trash".
        :param folder_name: str - name of the folder as saved in the Drive.
        :param is_name_subset: bool - True if name is a subset of the actual
        name of the folder, False otherwise.
        :return: dict - of folder ID and any associated children.
        """
        condition = f"mimeType='{self.google_mime_types['folder']}'"
        operator = "contains" if is_name_subset else "="
        folder_query = f"name {operator} '{folder_name}' and {condition}"
        logger.info(folder_query)
        try:
            results = self.get_file_metadata_by_query(folder_query)
            folder_id = self.get_folder_id(results)
            if folder_id:
                inputs_file_id = self.get_folder_contents(folder_id)
                if inputs_file_id:
                    return {"self_id": folder_id, "children": inputs_file_id}
                else:
                    return {"self_id": folder_id, "children": None}
            else:
                logger.info("No Folder found!")
                return {"self_id": None, "children": None}
        except errors.HttpError as error:
            logger.error(f"ERROR in get_folder: {error}")
            return {"self_id": None, "children": None}

    def is_file_in_folder(self, parent_folder_id, file_name, file_type,
                          is_name_subset=False):
        """
        Checks if a file is present in a given folder.
        :param parent_folder_id: str - ID of the parent folder.
        :param file_name: str - name of the file.
        :param file_type: str - MIME type of file to be searched.
        :param is_name_subset: bool - True if name is a subset of the actual
        name of the folder, False otherwise.
        :return: tuple: bool (whether the file was found), str (logging
        message).
        """
        match = False
        operator = "contains" if is_name_subset else "="
        mime_type = f"mimeType='{file_type}'"
        if file_type.strip().lower() == "google sheets":
            mime_type = f"mimeType='{self.google_mime_types['spreadsheet']}'"
        elif file_type.strip().lower() == "drive folder":
            mime_type = f"mimeType='{self.google_mime_types['folder']}'"
        elif file_type.strip().lower() == "xls":
            mime_type = f"mimeType='{self.excel_mime_types['xls']}'"
        elif file_type.strip().lower() == "xlsx":
            mime_type = f"mimeType='{self.excel_mime_types['xlsx']}'"
        query = f"name {operator} '{file_name}' and {mime_type}"
        if parent_folder_id:
            query = f"name {operator} '{file_name}' and {mime_type} and " \
                    f"'{parent_folder_id}' in parents"
        try:
            logger.info(f"\ncheck-file-query: {query}")
            results = self.get_file_metadata_by_query(query)
            items = self.get_folder_id(results, check_parents=False)
            s = "{} {} found"
            if items:
                logger.info(items)
                n = len(items)
                msg = s.format(n, "files") if n != 1 else s.format(n, "file")
                match = True
            else:
                msg = s.format(0, "files")
            return match, msg
        except errors.HttpError as error:
            msg = f"ERROR in is_file_in_folder: {error}"
            return False, msg

    def move_file(self, file_id: str, target_id: str):
        """
        Moves/associates a file to another Drive folder (target_id).
        :param file_id: str - ID of the file to move.
        :param target_id: str - ID of the folder in which the file will move.
        :return: Response if there was no HTTPError raised, otherwise None.
        """
        assert self.drive_service is not None
        try:
            files_object = self.drive_service.files()
            update_response = files_object.update(
                fileId=file_id,
                addParents=target_id).execute()
            return update_response
        except errors.HttpError as e:
            logger.error(f"Could not move file to the parent folder "
                         f"because: {e}")
            return None

    def create_spreadsheet(self, title: str, data_df, folder_id: str = None):
        """
        Creates a new Google Sheet. If another sheet of the same name exists,
        it will try to create another Sheet with a different suffix (e.g. -vN,
        where N is the version number).
        :param title: str - name of the file.
        :param data_df: DataFrame to be saved.
        :param folder_id: str - ID of the folder to save the file in.
        :return: Response if there was no Exception raised, otherwise None.
        """
        assert self.sheets_service is not None
        if not data_df.empty:
            from itertools import product
            from string import ascii_uppercase
            pre_exists, msg_ = self.is_file_in_folder(
                parent_folder_id=folder_id,
                file_name=title,
                file_type="google sheets")
            if pre_exists:
                logger.info(msg_)
                file_pattern_ = r"(.*)-v(\d+)"
                match_ = search(file_pattern_, title)
                groups_ = match_.groups() if match_ else []
                new_version = int(groups_[-1]) + 1 if groups_ else 0
                prefix = groups_[0] if groups_ else title
                title = f"{prefix}-v{new_version}"
            spreadsheet = {
                "properties": {"title": title}
            }
            try:
                sheet_columns = list(ascii_uppercase)
                data_columns = data_df.columns
                if len(data_columns) > 26:
                    more_columns = ["".join(c) for c in product(
                        ascii_uppercase, repeat=2)]
                    sheet_columns.extend(more_columns)
                start_column = sheet_columns[0]
                end_column = sheet_columns[len(data_columns) - 1]
                new_sheet = self.sheets_service.spreadsheets().create(
                    body=spreadsheet, fields='spreadsheetId').execute()
                new_sheet_id = new_sheet.get('spreadsheetId')
                if new_sheet_id:
                    logger.info(f"New Spreadsheet-ID: {new_sheet_id}")
                    move_response = self.move_file(new_sheet_id, folder_id)
                    logger.info(f"Response for move object: {move_response}")
                    logger.info(f"Saving data")
                    sheet_name = "Sheet1"
                    logger.info(f"(Sheet name: {sheet_name}, "
                                f"columns: [{start_column}:{end_column}])")
                    request = {
                        "id": new_sheet_id,
                        "range": f"{sheet_name}!{start_column}:{end_column}",
                        "body": {"majorDimension": "ROWS",
                                 "values": util.df_to_list(data_df)}
                    }
                    try:
                        new_sheet = self.sheets_service.spreadsheets() \
                            .values().update(
                            spreadsheetId=request["id"],
                            range=request["range"],
                            body=request["body"],
                            valueInputOption="RAW").execute()
                        return new_sheet
                    except errors.HttpError as e:
                        logger.error(
                            f"Sheet couldn't be updated because of:{e}")
                        return None
                else:
                    return None
            except errors.HttpError as error:
                logger.error(f"New Sheet couldn't be created because of:"
                             f" {error}")
                return None

    def create_folder(self, folder_name: str, target_folder: str):
        """
        Creates a new Google Drive folder inside another.
        :param folder_name: str - name of the new folder.
        :param target_folder: str - name of the parent folder for this new one.
        :return: str - ID of the new created folder if successful, otherwise
        None.
        """
        assert self.drive_service is not None
        new_folder_id, create_response = None, None
        logger.info(
            f"Creating Folder '{folder_name}' inside '{target_folder}'")
        selected_folders = self.get_folder(target_folder)
        logger.info(selected_folders)
        parent_folder_id = selected_folders["self_id"]
        try:
            folder_metadata = {
                "name": folder_name,
                "mimeType": self.google_mime_types["folder"],
                "parents": [parent_folder_id]
            }
            if parent_folder_id:
                logger.info(
                    f"\tSelf ID of Parent Folder: '{parent_folder_id}'")
                folder_exists, msg = self.is_file_in_folder(
                    parent_folder_id=parent_folder_id,
                    file_name=folder_name,
                    file_type="folder")
                logger.info(f"{msg}: {folder_exists}")
                if not folder_exists:
                    create_response = self.drive_service.files().create(
                        body=folder_metadata).execute()
            else:
                logger.info("\tCould not trace Parent Folder ID.")
                folder_metadata.pop("parents")
                create_response = self.drive_service.files().create(
                    body=folder_metadata).execute()
            if create_response:
                logger.info(create_response)
                new_folder_id = create_response.get("id")

        except errors.HttpError as e:
            logger.error(f"\nERROR: Folder couldn't be created because:\n{e}")
        finally:
            return new_folder_id

    def sheet_to_df_dict(self, spreadsheet_id, num_cols=-1, start=0):
        """
        Converts a Google Sheets file into a dict containing all the sheets and
        their values.
        :param spreadsheet_id: str - ID of the Google Sheets file.
        :param num_cols: int - number of columns to consider (0 being the
        first).
            `num_cols = -1` will take all the columns.
        :param start: int - index of the column to start reading from.
        :return: dict - with every key being the sheet-name, and the
        corresponding value being the DataFrame, if everything works out,
        otherwise an empty dict.
        """
        assert self.sheets_service is not None
        input_sheet_cols = ":".join(
            alphabets[start] + alphabets[start + num_cols - 1])
        spreadsheet_dataframes = dict()
        try:
            logger.info("Preparing request ...")
            request = self.sheets_service.spreadsheets().get(
                spreadsheetId=spreadsheet_id)
            logger.info("Executing request ...")
            response = request.execute()
            if response:
                sheets = response["sheets"]
                for sheet in sheets:
                    sheet_title = sheet["properties"]["title"]
                    _range = f"{sheet_title}!{input_sheet_cols}"
                    sheet_data = self.sheets_service.spreadsheets() \
                        .values().get(spreadsheetId=spreadsheet_id,
                                      range=_range).execute()
                    if sheet_data:
                        formatted_data = util.dict_to_df(sheet_data)
                        logger.info("Formatted DataFrame shape:")
                        logger.info(formatted_data.shape)
                        spreadsheet_dataframes[sheet_title] = formatted_data
            else:
                logger.info(f"Response couldn't be captured.")
        except errors.HttpError as error:
            logger.info(f"ERROR in sheet_to_df_dict: {error}")
        finally:
            return spreadsheet_dataframes

    def handle_inputs(self, folder, file_name="inputs", num_cols=-1,
                      start_col=0):
        """
        Converts Google Sheets file with `file_name` from a folder `user`
        into a
        dict of sheet-values.
        :param folder: str - name of the folder containing the file.
        :param file_name: str - name of the Google Sheets file.
        :param num_cols: number of columns to consider (0 being the first).
            `num_cols = -1` will take all the columns.
        :param start_col: int - index of the column to start reading from.
        :return: dict - with every key being the sheet-name, and the
        corresponding value being the DataFrame, if everything works out,
        otherwise an empty dict.
        """
        file_data = dict()
        if folder:
            folder_condition = f"mimeType = '" \
                               f"{self.google_mime_types.get('folder')}'"
            folder_query = f"name contains '{folder}' and {folder_condition}"
            logger.info(folder_query)
            try:
                results = self.get_file_metadata_by_query(folder_query)
                folder_id = self.get_folder_id(results)
                if folder_id:
                    if isinstance(folder_id, str):
                        inputs_file_id = self.get_folder_contents(folder_id)
                        logger.info(inputs_file_id)
                        _mimes = self.google_mime_types["spreadsheet"]
                        file = util.get_spreadsheets_from_files(inputs_file_id,
                                                                _mimes,
                                                                file_name)
                        if file:
                            sheet_id = file[0].get("id")
                            file_data = self.sheet_to_df_dict(sheet_id,
                                                              num_cols,
                                                              start_col)
                    logger.info(f"Multiple folder IDs found:")
                    logger.info(folder_id)
                else:
                    logger.info("No Folder found!")
            except errors.HttpError as error:
                logger.error(f"ERROR in handle_inputs: {error}")
                return None
        return file_data

    def get_csv_file_list(self):
        """
        Fetches a list of all the CSV files in the associated Google Drive.
        :return: list - of dicts containing file IDs and names, if success,
        otherwise an empty list.
        """
        return self.get_file_metadata_by_query("mimeType='text/csv'")

    def get_excel_file_list(self):
        """
        Fetches a list of all the Excel files in the associated Google Drive.
        :return: list - of dicts containing file IDs and names, if success,
        otherwise an empty list.
        """
        new_query = " or ".join([f"mimeType='{t}'" for t in
                                 self.excel_mime_types.values()])
        return self.get_file_metadata_by_query(new_query)

    def get_sheets_list(self):
        """
        Fetches a list of all the Google Sheets in the associated Google Drive.
        :return: list - of dicts containing file IDs and names, if success,
        otherwise an empty list.
        """
        new_query = f"mimeType = '{self.google_mime_types.get('spreadsheet')}'"
        return self.get_file_metadata_by_query(new_query)

    def get_unique_spreadsheets_list(self):
        """
        Fetches a list of all the unique Excel files and Google Sheets in the
        associated Google Drive.
        :return: list - of dicts containing file IDs and names, if success,
        otherwise an empty list.
        """
        sheets = list(self.excel_mime_types.values())
        sheets += [self.google_mime_types.get("spreadsheet")]
        new_query = " or ".join([f"mimeType='{t}'" for t in sheets])
        _files = self.get_file_metadata_by_query(new_query)
        unique_file_names = list()
        unique_files = list()
        for _file in _files:
            _name = _file.get('name')
            if _name not in unique_file_names:
                unique_file_names.append(_name)
                unique_files.append(_file)
        return unique_files

    def get_parent_folder(self, file_name: str):
        """
        Returns the relevant data for parent(s) of the file with `file_id`.
        :param file_name: str - name of the file.
        :return: list - of data of parent(s) of the given file.
        """
        parents = list()
        try:
            query = f"name = '{file_name}'"
            response = self.get_file_metadata_by_query(query)
            for parent in response:
                _ids = parent.get("parents", [])
                _data = list()
                for _id in _ids:
                    _meta = self.get_file_metadata_by_id(_id)
                    _data.append(_meta)
                parents.append(_data)
        except errors.HttpError as err:
            logger.error(f"ERROR in <get_parent_folder>: encountered: {err}")
        finally:
            return parents

    def get_file_type_from_mime(self, mime_type: str):
        """
        Get MIME type of the file from its metadata.
        :param mime_type: string value of the MIME type
        :return: returns the MIME type from the lookup dict associated with
        the object, if it's valid, otherwise None.
        """
        all_mime_types = dict()
        all_mime_types.update(self.excel_mime_types)
        all_mime_types.update(self.google_mime_types)
        reversed_map = {v: k for k, v in all_mime_types.items()}
        return reversed_map.get(mime_type)

    def download_file(self, file_id: str):
        """
        Downloads a file stored on Google Drive into a MediaIoBaseDownload
        object (access the file data as
        ``MediaIoBaseDownload._fd``)
        :param file_id: str - ID of the file
        :return: googleapiclient.http.MediaIoBaseDownload object
        """
        _request = self.drive_service.files().get_media(fileId=file_id)
        buffered_io = BytesIO()
        downloader = MediaIoBaseDownload(buffered_io, _request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download progress: {int(status.progress() * 100)}")
        return downloader

    def download_workspace_doc(self, file_id: str, mime_type: str):
        """
        Downloads a Google Workspace Document on Google Drive into a
        MediaIoBaseDownload object (access the file data as
        ``MediaIoBaseDownload._fd``)
        :param file_id: str - ID of the file
        :param mime_type: str - MIME type of the Google Workspace Document
        :return: googleapiclient.http.MediaIoBaseDownload object
        """
        _request = self.drive_service.files().export_media(fileId=file_id,
                                                           mimeType=mime_type)
        buffered_io = BytesIO()
        downloader = MediaIoBaseDownload(buffered_io, _request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download progress: {int(status.progress() * 100)}")
        return downloader
