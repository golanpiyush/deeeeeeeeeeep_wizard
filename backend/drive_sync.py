"""
drive_sync.py
==============
Read-only access to the team's shared Google Drive folder (the one with
lunar_data / ohrc_2024 / etc.) via the official Google Drive API + OAuth2.

WHY THIS APPROACH, NOT USERNAME/PASSWORD:
    Google blocks plain username/password sign-in for scripts like this,
    and even if it didn't, putting your real Gmail password in a .env
    file is a real security risk -- one accidental `git add .` away from
    leaking it into your public GitHub repo. OAuth2 never touches your
    password at all: you log in once through a real Google browser
    popup, and a local token file (NOT your password) is cached for
    every run after that.

SETUP (one-time):
    1. Get credentials.json from Google Cloud Console (OAuth client ID,
       "Desktop app" type).
    2. Put credentials.json in this same folder.
    3. Run this script once -- a browser window opens, you log in and
       approve access, and a token.json gets cached locally. Every run
       after that is silent (no browser popup) until the token expires.
    4. Add BOTH credentials.json and token.json to .gitignore -- neither
       should ever be committed, even though neither contains your
       actual password.

USAGE:
    # Stream a file's contents directly into memory (default, no disk write):
    python drive_sync.py --list                          # see what's in the folder
    python drive_sync.py --fetch "lunar_crop.png"          # stream one file by name

    # Force saving to disk instead of streaming (for big files you'll
    # reuse repeatedly, e.g. the 1.2GB .img files):
    python drive_sync.py --fetch "ch2_ohr_..._d_img_d18.img" --download-first

    # Search by pattern (matches anywhere in the filename):
    python drive_sync.py --search ".xml"

SCOPE: this script is READ-ONLY (drive.readonly scope) -- it cannot
create, edit, or delete anything in the Drive, by design.
"""

from __future__ import annotations

import argparse
import io
import os
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(THIS_DIR, "credentials.json")
TOKEN_PATH = os.path.join(THIS_DIR, "token.json")

DOWNLOAD_DIR = os.path.join(THIS_DIR, "downloaded")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def get_drive_service():
    """
    Returns an authenticated Drive API client.

    First run: opens a browser window for you to log in and approve
    read-only access, then caches the resulting token to token.json.
    Every run after that reuses the cached token silently -- no browser
    popup -- until it expires, at which point it refreshes automatically.
    """
    if not os.path.exists(CREDENTIALS_PATH):
        print(
            f"ERROR: {CREDENTIALS_PATH} not found.\n\n"
            "You need to download this from Google Cloud Console first:\n"
            "  1. console.cloud.google.com -> APIs & Services -> Credentials\n"
            "  2. Create Credentials -> OAuth client ID -> Desktop app\n"
            "  3. Download the JSON, save it here as 'credentials.json'\n"
        )
        sys.exit(1)

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Listing / searching
# ---------------------------------------------------------------------------


def list_files(service, query: str | None = None, page_size: int = 50) -> list[dict]:
    """
    Lists files visible to this account, optionally filtered by a Drive
    API search query. Returns a list of {id, name, mimeType, size} dicts.
    """
    q = query or "trashed = false"
    results = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=q,
                pageSize=page_size,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )
        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return results


def search_by_name_fragment(service, fragment: str) -> list[dict]:
    """Finds files whose name CONTAINS the given fragment."""
    query = f"name contains '{fragment}' and trashed = false"
    return list_files(service, query=query)


def find_exact_name(service, name: str) -> dict | None:
    """Finds a single file by exact name match."""
    query = f"name = '{name}' and trashed = false"
    matches = list_files(service, query=query)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"WARNING: {len(matches)} files named exactly '{name}' found -- using the first one.")
        for m in matches:
            print(f"    id={m['id']}  modified={m.get('modifiedTime')}")
    return matches[0]


# ---------------------------------------------------------------------------
# Fetching (stream vs download)
# ---------------------------------------------------------------------------


def stream_file_bytes(service, file_id: str) -> bytes:
    """
    Fetches a file's full contents directly into memory as bytes -- no
    disk write. Fine for small-to-medium files. For very large files
    (the 1.2GB .img products), prefer download_file_to_disk() instead.
    """
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    buffer.seek(0)
    return buffer.read()


def download_file_to_disk(service, file_id: str, filename: str, out_dir: str = DOWNLOAD_DIR) -> str:
    """
    Downloads a file to disk (chunked, so large files don't need to fit
    in memory during the download), returns the local path.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    request = service.files().get_media(fileId=file_id)
    with open(out_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"\r    Downloading... {pct}%", end="", flush=True)
    print()

    return out_path


def fetch_file(service, name: str, download_first: bool = False):
    """
    High-level entry point: finds a file by exact name, then either
    streams its bytes into memory (default) or downloads it to disk
    first and returns the local path (if download_first=True).
    """
    file_info = find_exact_name(service, name)
    if file_info is None:
        raise FileNotFoundError(f"No file named '{name}' found in Drive.")

    size_bytes = int(file_info.get("size", 0) or 0)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Found: {file_info['name']}  ({size_mb:.1f} MB, id={file_info['id']})")

    if download_first:
        print("Mode: download-first (saving to disk)...")
        path = download_file_to_disk(service, file_info["id"], file_info["name"])
        print(f"Saved -> {path}")
        return path
    else:
        print("Mode: stream (fetching into memory, no disk write)...")
        if size_mb > 200:
            print(
                f"    NOTE: this file is {size_mb:.0f} MB -- streaming this into memory "
                f"repeatedly is slow/wasteful. Consider --download-first for files this large."
            )
        data = stream_file_bytes(service, file_info["id"])
        print(f"Fetched {len(data):,} bytes into memory.")
        return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Read-only access to the team's shared Google Drive folder."
    )
    parser.add_argument("--list", action="store_true", help="List all visible files.")
    parser.add_argument("--search", metavar="FRAGMENT", help="Search filenames containing FRAGMENT.")
    parser.add_argument("--fetch", metavar="NAME", help="Fetch a file by exact name.")
    parser.add_argument(
        "--download-first",
        action="store_true",
        help="Save the fetched file to disk instead of streaming it into memory.",
    )
    args = parser.parse_args()

    if not any([args.list, args.search, args.fetch]):
        parser.print_help()
        return

    print("Authenticating with Google Drive (first run opens a browser window)...")
    service = get_drive_service()
    print("Authenticated.\n")

    if args.list:
        files = list_files(service)
        print(f"{len(files)} file(s) visible:\n")
        for f in files:
            size = f.get("size")
            size_str = f"{int(size)/1024:.1f} KB" if size else "(folder or unknown size)"
            print(f"  {f['name']:<55} {size_str:>15}   [{f['mimeType']}]")

    elif args.search:
        matches = search_by_name_fragment(service, args.search)
        print(f"{len(matches)} match(es) for '{args.search}':\n")
        for f in matches:
            size = f.get("size")
            size_str = f"{int(size)/1024:.1f} KB" if size else "(folder or unknown size)"
            print(f"  {f['name']:<55} {size_str:>15}")

    elif args.fetch:
        result = fetch_file(service, args.fetch, download_first=args.download_first)
        if isinstance(result, bytes):
            print(f"\nGot {len(result):,} bytes in memory. (Not saved to disk -- use --download-first to save.)")


if __name__ == "__main__":
    main()
