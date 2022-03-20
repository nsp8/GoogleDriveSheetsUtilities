# Google Drive & Sheets Utilities
Leveraging Google Drive and Sheets APIs to create folders, spreadsheets; read and save data.

---
## Prerequisites
Before executing anything, please go through the steps [here](https://developers.google.com/drive/api/v3/quickstart/python). It will kickstart the authorization process for your own app. \
Save the `credentials.json` file (not saved in this repo) generated from the first step, in the parent directory to enable Google Drive and Sheets capabilities.
> *Note*: the `credentials.json` file has to be present in order to work with the APIs.

> *Note*: If while creating srive/sheets service objects, you get `"error": "invalid_grant", "error_description": "Bad Request"`, the `credentials.json` might have been expired. 
> Go to the Google Cloud Console and generate a new OAuth Client ID (find more information on the [guide page](https://developers.google.com/workspace/guides/create-credentials#desktop-app) for your app).