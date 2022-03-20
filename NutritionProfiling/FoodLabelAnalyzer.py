import logging
import pandas as pd
from .. import google_drive

logger = logging.getLogger("food_label_analyzer")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("food_label_analyzer.log")
logger.addHandler(file_handler)
formatter = logging.Formatter("%(asctime)s: "
                              "%(levelname)s: "
                              "%(name)s: "
                              "%(message)s")
file_handler.setFormatter(formatter)
drive_api = google_drive.GoogleDriveAPI()


def read_sheet_data(file_name, sheet_name=None, skip_rows=0):
    try:
        folder_name = "Nutrition Profiling"
        folder = drive_api.get_folder(folder_name)
        file_object = [f for f in folder["children"]
                       if not f["trashed"] and f["name"] == file_name][0]
        file_contents = drive_api.download_file(file_object["id"],
                                                drive_api.drive_service)

        if sheet_name:
            file_data_all = pd.read_excel(file_contents._fd,
                                          sheet_name=sheet_name,
                                          skiprows=skip_rows)
            logger.info(f"Size of DataFrame: {file_data_all.shape}")
        else:
            file_data_all = pd.read_excel(file_contents._fd,
                                          sheet_name=sheet_name)
            logger.info(f"Number of sheets in file: {len(file_data_all)}")
        return file_data_all
    except Exception as e:
        logger.error(f"Error while reading file from Google Drive: {e}")


def process_file_data(sheet_df):
    file_data_cleaner = sheet_df.dropna(how="all")
    # TODO: Parse DataFrame for obtaining parameter values given a product name
