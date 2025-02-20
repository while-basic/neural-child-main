"""
obsidian_api.py

A Python module to interact with the Obsidian Local REST API.
This module wraps common endpoints (active file, vault files, periodic notes, commands, etc.)
into methods of the ObsidianAPI class.
"""

import requests
from typing import Optional, Dict, Any, List

class ObsidianAPI:
    def __init__(self, api_key: str, base_url: str = "http://127.0.0.1:27123"):
        """
        Initialize the Obsidian API client.

        Args:
            api_key (str): Your API key from Obsidian.
            base_url (str): Base URL of the API. Default is "http://127.0.0.1:27123".
        """
        if not api_key:
            raise ValueError("An API key is required.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Default headers. Note that some endpoints may override Content-Type.
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "text/markdown"
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
        json_data: Any = None,
    ) -> requests.Response:
        """
        Internal helper to perform an HTTP request.

        Args:
            method (str): HTTP method (GET, POST, PUT, PATCH, DELETE).
            endpoint (str): API endpoint (e.g. "/vault/Note1.md").
            headers (dict, optional): Additional headers.
            params (dict, optional): Query parameters.
            data: Raw body data.
            json_data: JSON body data.

        Returns:
            requests.Response: The response object.

        Raises:
            Exception: If the request returns a non-200/204 status code.
        """
        url = f"{self.base_url}{endpoint}"
        req_headers = self.headers.copy()
        if headers:
            req_headers.update(headers)
        response = requests.request(method, url, headers=req_headers, params=params, data=data, json=json_data)
        if response.status_code not in (200, 204):
            raise Exception(f"Error {response.status_code} on {method} {url}: {response.text}")
        return response

    # ----- Server & Active File Endpoints -----

    def get_server_info(self) -> Dict[str, Any]:
        """
        Returns basic details about the server.
        """
        response = self._request("GET", "/")
        return response.json()

    def get_active_file(self, as_json: bool = False) -> Any:
        """
        Returns the content of the active file.

        Args:
            as_json (bool): If True, sets the Accept header to get a JSON representation.

        Returns:
            str or dict: The note content as raw text or as a parsed JSON object.
        """
        headers = {}
        if as_json:
            headers["Accept"] = "application/vnd.olrapi.note+json"
        response = self._request("GET", "/active/", headers=headers)
        return response.json() if as_json else response.text

    def delete_active_file(self):
        """
        Deletes the currently active file.
        """
        self._request("DELETE", "/active/")

    def patch_active_file(
        self,
        operation: str,
        target_type: str,
        target: str,
        content: Any,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Inserts content into the active file relative to a heading, block, or frontmatter field.

        Args:
            operation (str): "append", "prepend", or "replace".
            target_type (str): "heading", "block", or "frontmatter".
            target (str): The target to patch.
            content (Any): Content to insert.
            extra_headers (dict, optional): Additional headers (e.g. Target-Delimiter).

        Returns:
            dict or None: The JSON response if any.
        """
        headers = {
            "Operation": operation,
            "Target-Type": target_type,
            "Target": target,
        }
        if extra_headers:
            headers.update(extra_headers)
        response = self._request("PATCH", "/active/", headers=headers, data=content)
        return response.json() if response.content else None

    def append_to_active_file(self, content: str):
        """
        Appends content to the end of the active file.
        """
        self._request("POST", "/active/", data=content)

    def update_active_file(self, content: str):
        """
        Replaces the content of the active file.
        """
        self._request("PUT", "/active/", data=content)

    # ----- Commands Endpoints -----

    def get_commands(self) -> List[Dict[str, Any]]:
        """
        Retrieves a list of available commands.
        """
        response = self._request("GET", "/commands/")
        data = response.json()
        return data.get("commands", [])

    def execute_command(self, command_id: str):
        """
        Executes a specified command.

        Args:
            command_id (str): The ID of the command to execute.
        """
        endpoint = f"/commands/{command_id}/"
        self._request("POST", endpoint)

    # ----- Open File Endpoint -----

    def open_file(self, filename: str, new_leaf: bool = False):
        """
        Opens the specified file in Obsidian. If the file does not exist, it will be created.

        Args:
            filename (str): The relative path to the file.
            new_leaf (bool): Whether to open in a new leaf.
        """
        params = {"newLeaf": str(new_leaf).lower()}
        endpoint = f"/open/{filename}"
        self._request("POST", endpoint, params=params)

    # ----- Periodic Notes Endpoints -----

    def get_periodic_note(self, period: str = "daily", as_json: bool = True) -> Any:
        """
        Retrieves the periodic note for the specified period.

        Args:
            period (str): One of "daily", "weekly", "monthly", "quarterly", or "yearly".
            as_json (bool): If True, returns a JSON representation.

        Returns:
            dict or str: The note as JSON or raw text.
        """
        endpoint = f"/periodic/{period}/"
        headers = {}
        if as_json:
            headers["Accept"] = "application/vnd.olrapi.note+json"
        response = self._request("GET", endpoint, headers=headers)
        return response.json() if as_json else response.text

    def delete_periodic_note(self, period: str = "daily"):
        """
        Deletes the periodic note for the specified period.
        """
        endpoint = f"/periodic/{period}/"
        self._request("DELETE", endpoint)

    def patch_periodic_note(
        self,
        period: str,
        operation: str,
        target_type: str,
        target: str,
        content: Any,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Patches the periodic note by inserting content relative to a target.

        Args:
            period (str): The period (e.g., "daily").
            operation (str): "append", "prepend", or "replace".
            target_type (str): "heading", "block", or "frontmatter".
            target (str): The target within the note.
            content (Any): The content to insert.
            extra_headers (dict, optional): Additional headers if needed.

        Returns:
            dict or None: The JSON response if any.
        """
        endpoint = f"/periodic/{period}/"
        headers = {
            "Operation": operation,
            "Target-Type": target_type,
            "Target": target,
        }
        if extra_headers:
            headers.update(extra_headers)
        response = self._request("PATCH", endpoint, headers=headers, data=content)
        return response.json() if response.content else None

    def append_to_periodic_note(self, period: str, content: str):
        """
        Appends content to the periodic note for the specified period.
        """
        endpoint = f"/periodic/{period}/"
        self._request("POST", endpoint, data=content)

    def update_periodic_note(self, period: str, content: str):
        """
        Replaces the content of the periodic note for the specified period.
        """
        endpoint = f"/periodic/{period}/"
        self._request("PUT", endpoint, data=content)

    # ----- Vault Files Endpoints -----

    def list_vault_files(self, path: str = "") -> List[str]:
        """
        Lists files in the specified directory (or root if path is empty).

        Args:
            path (str): Relative path to the directory in the vault.

        Returns:
            list: A list of file and directory names.
        """
        endpoint = f"/vault/{path}" if path else "/vault/"
        response = self._request("GET", endpoint)
        data = response.json()
        return data.get("files", [])

    def delete_vault_file(self, filename: str):
        """
        Deletes a file from the vault.

        Args:
            filename (str): Relative path to the file.
        """
        endpoint = f"/vault/{filename}"
        self._request("DELETE", endpoint)

    def get_vault_file(self, filename: str, as_json: bool = False) -> Any:
        """
        Retrieves the content of a vault file.

        Args:
            filename (str): Relative path to the file.
            as_json (bool): If True, returns a JSON representation.

        Returns:
            dict or str: The file as JSON or raw text.
        """
        headers = {}
        if as_json:
            headers["Accept"] = "application/vnd.olrapi.note+json"
        endpoint = f"/vault/{filename}"
        response = self._request("GET", endpoint, headers=headers)
        return response.json() if as_json else response.text

    def patch_vault_file(
        self,
        filename: str,
        operation: str,
        target_type: str,
        target: str,
        content: Any,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Inserts content into a vault file relative to a target.

        Args:
            filename (str): Relative path to the file.
            operation (str): "append", "prepend", or "replace".
            target_type (str): "heading", "block", or "frontmatter".
            target (str): The target within the note.
            content (Any): The content to insert.
            extra_headers (dict, optional): Additional headers if needed.

        Returns:
            dict or None: The JSON response if any.
        """
        headers = {
            "Operation": operation,
            "Target-Type": target_type,
            "Target": target,
        }
        if extra_headers:
            headers.update(extra_headers)
        endpoint = f"/vault/{filename}"
        response = self._request("PATCH", endpoint, headers=headers, data=content)
        return response.json() if response.content else None

    def append_to_vault_file(self, filename: str, content: str):
        """
        Appends content to an existing vault file (or creates it if missing).
        """
        endpoint = f"/vault/{filename}"
        self._request("POST", endpoint, data=content)

    def put_vault_file(self, filename: str, content: str):
        """
        Creates a new file or updates an existing one in the vault.

        Args:
            filename (str): Relative path to the file.
            content (str): The markdown content.
        """
        endpoint = f"/vault/{filename}"
        self._request("PUT", endpoint, data=content)

    # ----- Search Endpoints -----

    def search(self, query: Dict[str, Any]) -> Any:
        """
        Performs a search using the /search/ endpoint.

        Args:
            query (dict): A dictionary representing the search query.

        Returns:
            dict: The search results.
        """
        endpoint = "/search/"
        response = self._request("POST", endpoint, json_data=query)
        return response.json()

    def simple_search(self, query: str) -> Any:
        """
        Performs a simple text search using the /search/simple/ endpoint.

        Args:
            query (str): The text query.

        Returns:
            dict: The search results.
        """
        endpoint = "/search/simple/"
        response = self._request("POST", endpoint, data=query)
        return response.json()

# --- Example Usage ---
# To use this module in another script, you could do:
#
#   from obsidian_api import ObsidianAPI
#
#   API_KEY = "a2af645c9e4b5ce13c8d023f4e318b39f6855afec998f2b8f38a201d232e7e6c"
#   obs_api = ObsidianAPI(api_key=API_KEY)
#
#   # Create or update a note:
#   content = "# My New Note\n\nThis note links to [[AnotherNote]]."
#   obs_api.put_vault_file("MyNewNote.md", content)
#
#   # Get server information:
#   info = obs_api.get_server_info()
#   print(info)
#
#   # List files in the vault root:
#   files = obs_api.list_vault_files()
#   print(files)
#
# This module is designed to be extensible so you can add or modify methods as needed.

# obsidiantest.py
from obsidian_api import ObsidianAPI

API_KEY = "a2af645c9e4b5ce13c8d023f4e318b39f6855afec998f2b8f38a201d232e7e6c"
obs_api = ObsidianAPI(api_key=API_KEY)

# For example, create a new note:
note_content = "# Note Example\n\nThis note links to [[AnotherNote]]."
obs_api.put_vault_file("NoteExample.md", note_content)


PS E:\python-projects\local_mind> python .\obsidiantest.py
Traceback (most recent call last):
  File "E:\python-projects\local_mind\obsidiantest.py", line 1, in <module>
    from obsidian_api import ObsidianAPI
ImportError: cannot import name 'ObsidianAPI' from 'obsidian_api' (E:\python-projects\local_mind\obsidian_api.py)