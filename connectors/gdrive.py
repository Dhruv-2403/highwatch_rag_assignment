
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

SUPPORTED_MIME = {
    "application/pdf",
    "application/vnd.google-apps.document",
    "text/plain",
}
EXPORT_MIME = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
}


@dataclass
class DriveFile:

    file_id: str
    file_name: str
    mime_type: str
    content: bytes
    web_view_link: str = ""
    modified_time: str = ""
    extra_metadata: dict = field(default_factory=dict)


class GoogleDriveConnector:


    def __init__(self) -> None:
        self._service = None


    def _get_credentials(self) -> Credentials:
        method = settings.google_auth_method.lower()

        if method == "service_account":
            return service_account.Credentials.from_service_account_file(
                str(settings.google_service_account_file),
                scopes=SCOPES,
            )

 
        creds: Credentials | None = None
        token_path = settings.google_token_file

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(settings.google_oauth_client_secret_file),
                    SCOPES,
                )
                creds = flow.run_local_server(port=0)

            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())

        return creds

    def _build_service(self):
        if self._service is None:
            creds = self._get_credentials()
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

 
    def _list_files(self, folder_ids: list[str] | None = None) -> list[dict]:
    
        service = self._build_service()
        mime_filter = " or ".join(
            f"mimeType='{m}'" for m in SUPPORTED_MIME
        )

        if folder_ids:
  
            folder_filter = " or ".join(
                f"'{fid}' in parents" for fid in folder_ids
            )
            q = f"({folder_filter}) and ({mime_filter}) and trashed=false"
        else:
            q = f"({mime_filter}) and trashed=false"

        results: list[dict] = []
        page_token: str | None = None

        while True:
            resp = (
                service.files()
                .list(
                    q=q,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, webViewLink, modifiedTime)",
                    pageToken=page_token,
                    pageSize=100,
                )
                .execute()
            )
            results.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        logger.info("Found %d files in Drive", len(results))
        return results

    def _download_file(self, file_meta: dict) -> bytes | None:
     
        service = self._build_service()
        fid = file_meta["id"]
        mime = file_meta["mimeType"]

        try:
            if mime in EXPORT_MIME:
              
                export_mime = EXPORT_MIME[mime]
                request = service.files().export_media(fileId=fid, mimeType=export_mime)
            else:
                request = service.files().get_media(fileId=fid)

            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buf.getvalue()

        except Exception as exc:
            logger.warning("Failed to download %s (%s): %s", file_meta["name"], fid, exc)
            return None

    
    def fetch_files(self, folder_ids: list[str] | None = None) -> Iterator[DriveFile]:
  
        folder_ids = folder_ids or settings.gdrive_folder_id_list or None
        files = self._list_files(folder_ids)

        for meta in files:
            mime = meta["mimeType"]
            content = self._download_file(meta)
            if content is None:
                continue


            effective_mime = EXPORT_MIME.get(mime, mime)

            yield DriveFile(
                file_id=meta["id"],
                file_name=meta["name"],
                mime_type=effective_mime,
                content=content,
                web_view_link=meta.get("webViewLink", ""),
                modified_time=meta.get("modifiedTime", ""),
            )
            logger.debug("Fetched: %s", meta["name"])
