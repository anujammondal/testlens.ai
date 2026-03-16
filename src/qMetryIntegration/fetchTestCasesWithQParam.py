import requests
import json
import time
import os
from pathlib import Path

# Load environment variables from .env file in project root
try:
    from dotenv import load_dotenv
    # Find project root (2 levels up from this file: src/qMetryIntegration/fetchTestCases.py)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system environment variables

# ---------------- CONFIG ----------------
# QMetry Cloud API base URL
QMETRY_API_URL = os.getenv("QMETRY_API_URL", "https://qtmcloud.qmetry.com/rest/api/latest")

# Project settings
PROJECT_ID = os.getenv("QMETRY_PROJECT_ID", "10081")  # GQA project
PROJECT_KEY = os.getenv("QMETRY_PROJECT_KEY", "GQA")

def get_folder_id():
    """Get folder ID from QMETRY_FOLDER_ID environment variable only."""
    return os.getenv("QMETRY_FOLDER_ID", "")

MAX_RESULTS = 100  # Results per page
RETRY_COUNT = 5
BACKOFF_FACTOR = 2

OUTPUT_FILE = "qmetry_testcases.json"


def get_api_key():
    """Get API key from environment variable."""
    return os.getenv("QMETRY_API_KEY", "")


def get_headers():
    """Build headers with API key."""
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "apiKey": get_api_key()
    }


# ---------------- RETRY LOGIC ----------------
def request_with_retry(method, url, **kwargs):
    for attempt in range(1, RETRY_COUNT + 1):
        response = requests.request(method, url, **kwargs)

        if response.status_code < 400:
            return response

        if response.status_code in [429, 500, 502, 503, 504]:
            wait = BACKOFF_FACTOR ** attempt
            print(f"⏱ Retry {attempt} for {url} after {wait}s")
            time.sleep(wait)
        else:
            print(f"❌ HTTP {response.status_code} Error")
            print(f"URL: {url}")
            print(f"Response Body: {response.text[:1000]}")
            response.raise_for_status()

    raise Exception(f"❌ Failed after {RETRY_COUNT} retries: {url}")


# ---------------- GET FOLDER INFO ----------------
def get_folder_info(folder_id):
    """
    Get folder name and path for a given folder ID.
    
    Uses the /projects/{PROJECT_ID}/testcase-folders/{folder_id} endpoint.
    
    Args:
        folder_id: The folder ID
    
    Returns:
        Dict with 'id' and 'name' for the folder
    """
    if not folder_id:
        return {"id": None, "name": None}
    
    try:
        url = f"{QMETRY_API_URL}/projects/{PROJECT_ID}/testcase-folders/{folder_id}"
        response = request_with_retry("GET", url, headers=get_headers())
        data = response.json()
        return {
            "id": str(data.get("id", folder_id)),
            "name": data.get("name", "Unknown")
        }
    except Exception:
        return {"id": str(folder_id), "name": "Unknown"}


# ---------------- GET CHILD FOLDER IDS ----------------
def get_child_folder_ids(parent_folder_id):
    """
    Get all child folder IDs (recursively) for a given parent folder.
    
    Uses the /projects/{PROJECT_ID}/testcase-folders endpoint to get the folder structure,
    then finds all descendants of the specified parent folder.
    
    Args:
        parent_folder_id: The parent folder ID
    
    Returns:
        List of dicts with 'id' and 'name' for each child folder
    """
    url = f"{QMETRY_API_URL}/projects/{PROJECT_ID}/testcase-folders"
    response = request_with_retry("GET", url, headers=get_headers())
    folders = response.json()
    
    def find_children_recursive(folders_list, target_id, collect=False):
        """Recursively find all children of target folder."""
        children = []
        
        for folder in folders_list:
            fid = folder.get("id")
            fname = folder.get("name", "Unknown")
            
            if fid == target_id:
                # Found target, now collect all its children
                for child in folder.get("children", []):
                    children.append({"id": child.get("id"), "name": child.get("name", "Unknown")})
                    # Recursively get grandchildren
                    children.extend(find_children_recursive([child], None, collect=True))
            elif collect:
                # We're collecting children of a parent
                for child in folder.get("children", []):
                    children.append({"id": child.get("id"), "name": child.get("name", "Unknown")})
                    children.extend(find_children_recursive([child], None, collect=True))
            else:
                # Keep searching in children
                children.extend(find_children_recursive(folder.get("children", []), target_id, collect=False))
        
        return children
    
    return find_children_recursive(folders.get("data", []), int(parent_folder_id))


# ---------------- FETCH TEST CASES FROM FOLDER ----------------
def fetch_test_cases_from_folder(folder_id, folder_name="", with_child=False):
    """
    Fetch all test cases from a specific folder.
    
    Args:
        folder_id: The folder ID to fetch from
        folder_name: Name of the folder (for labeling)
        with_child: Whether to include child folders
    
    Returns:
        List of test cases
    """
    all_testcases = []
    start_at = 0
    base_url = f"{QMETRY_API_URL}/testcases/search"
    
    while True:
        filter_payload = {
            "projectId": str(PROJECT_ID),
            "folderId": str(folder_id),
            "withChild": with_child
        }
        
        payload = {
            "filter": filter_payload,
            "startAt": start_at,
            "maxResults": MAX_RESULTS
        }
        
        response = request_with_retry(
            "POST", base_url, headers=get_headers(), json=payload
        )
        
        data = response.json()
        testcases = data.get("data", [])
        total = data.get("total", 0)
        
        # Add folder info to each test case
        for tc in testcases:
            tc["folder"] = {"id": str(folder_id), "name": folder_name}
        
        all_testcases.extend(testcases)
        
        if not testcases or len(all_testcases) >= total:
            break
        
        start_at += MAX_RESULTS
        time.sleep(0.05)
    
    return all_testcases


# ---------------- FETCH TEST CASES WITH FOLDER TRAVERSAL ----------------
def fetch_test_cases_by_folder(parent_folder_id, parent_folder_name="Root"):
    """
    Fetch all test cases from a folder and its children.
    Uses withChild=True to get all test cases, then fetches folder info for each.
    
    Returns a dict with folder info and all test cases.
    """
    result = {
        "parentFolder": {"id": str(parent_folder_id), "name": parent_folder_name},
        "folders": [],
        "testcases": []
    }
    
    # First, get test cases from the parent folder itself (without children)
    print(f"\n📂 Fetching from parent folder: {parent_folder_name} ({parent_folder_id})")
    parent_tcs = fetch_test_cases_from_folder(parent_folder_id, parent_folder_name, with_child=False)
    
    if parent_tcs:
        result["folders"].append({
            "id": str(parent_folder_id),
            "name": parent_folder_name,
            "testcaseCount": len(parent_tcs)
        })
        result["testcases"].extend(parent_tcs)
        print(f"   ✅ Found {len(parent_tcs)} test cases")
    else:
        print(f"   📭 No test cases in parent folder")
    
    # Fetch all test cases from child folders using withChild=True
    print(f"\n🔍 Fetching test cases from child folders...")
    all_tcs = fetch_test_cases_from_folder(parent_folder_id, "", with_child=True)
    
    # Remove duplicates (parent folder TCs already added)
    existing_ids = {tc.get("id") for tc in result["testcases"]}
    new_tcs = [tc for tc in all_tcs if tc.get("id") not in existing_ids]
    
    if not new_tcs:
        print(f"   📭 No additional test cases in child folders")
        print(f"\n📊 Total: {len(result['testcases'])} test cases from {len(result['folders'])} folders")
        return result
    
    # Fetch folder info for each test case using detail endpoint
    print(f"\n🔄 Fetching folder info for {len(new_tcs)} test cases...")
    folders_map = {}
    
    for i, tc in enumerate(new_tcs, 1):
        tc_id = tc.get("id")
        version_no = tc.get("version", {}).get("versionNo", 1)
        
        try:
            url = f"{QMETRY_API_URL}/testcases/{tc_id}/versions/{version_no}"
            response = request_with_retry("GET", url, headers=get_headers())
            details = response.json().get("data", {})
            
            # Get folder info from details
            folder_info = details.get("folder", {})
            folder_id = folder_info.get("id")
            folder_name = folder_info.get("name", "Unknown")
            
            if folder_id:
                tc["folder"] = {"id": str(folder_id), "name": folder_name}
                if folder_id not in folders_map:
                    folders_map[folder_id] = {"id": str(folder_id), "name": folder_name, "testcaseCount": 0}
                folders_map[folder_id]["testcaseCount"] += 1
                
        except Exception as e:
            pass  # Skip if we can't get details
        
        if i % 20 == 0 or i == len(new_tcs):
            print(f"   ⚙️ Processed {i}/{len(new_tcs)} test cases")
        
        time.sleep(0.02)
    
    result["testcases"].extend(new_tcs)
    result["folders"].extend(folders_map.values())
    
    print(f"\n📊 Total: {len(result['testcases'])} test cases from {len(result['folders'])} folders")
    return result


# ---------------- FETCH TEST CASES (SIMPLE) ----------------
def fetch_test_cases(limit=None, folder_id=None, with_child=True, traverse_folders=False):
    """
    Main function to fetch test cases.
    
    Args:
        limit: Maximum number of test cases to fetch
        folder_id: Folder ID to fetch from
        with_child: Include test cases from child folders
        traverse_folders: If True, traverse each child folder and map test cases to folders
    
    Returns:
        Dict with parentFolder info and testcases list
    
    Note: QMetry Cloud API has a known limitation where pagination with withChild=True
    may not work correctly - it often returns only 50 items regardless of the actual count.
    
    WORKAROUND: When with_child=True and a folder_id is specified, this function:
    1. Gets the folder structure to find all child folder IDs
    2. Queries each child folder individually (without withChild)
    3. Combines and deduplicates the results
    
    This workaround reliably fetches ALL test cases from a folder hierarchy.
    """
    if traverse_folders and folder_id:
        return fetch_test_cases_by_folder(folder_id, "Root Folder")
    
    seen_ids = set()
    all_testcases = []
    base_url = f"{QMETRY_API_URL}/testcases/search"
    
    # WORKAROUND for QMetry pagination bug:
    # When with_child=True, query each child folder individually instead
    if folder_id and with_child:
        print(f"📂 Using child folder workaround for pagination bug...")
        
        # Get child folder IDs
        child_folders = get_child_folder_ids(folder_id)
        print(f"   Found {len(child_folders)} child folders")
        
        # Include the parent folder itself
        folders_to_query = [{"id": int(folder_id), "name": "Parent"}] + child_folders
        
        for folder_info in folders_to_query:
            fid = folder_info["id"]
            fname = folder_info["name"]
            
            payload = {
                "filter": {
                    "projectId": str(PROJECT_ID),
                    "folderId": str(fid),
                    "withChild": False  # Query each folder directly, no children
                },
                "startAt": 0,
                "maxResults": MAX_RESULTS
            }
            
            response = request_with_retry("POST", base_url, headers=get_headers(), json=payload)
            data = response.json()
            testcases = data.get("data", [])
            total = data.get("total", 0)
            
            # Add folder info and deduplicate
            new_count = 0
            for tc in testcases:
                tc_id = tc.get("id")
                if tc_id and tc_id not in seen_ids:
                    seen_ids.add(tc_id)
                    tc["folder"] = {"id": str(fid), "name": fname}
                    all_testcases.append(tc)
                    new_count += 1
            
            if total > 0:
                print(f"   📁 {fname}: {new_count} test cases")
            
            if limit and len(all_testcases) >= limit:
                all_testcases = all_testcases[:limit]
                break
            
            time.sleep(0.05)
        
        print(f"📥 Total: {len(all_testcases)} unique test cases from {len(folders_to_query)} folders")
    
    else:
        # Standard fetch (no folder or no children)
        start_at = 0
        
        # First, get the total count
        filter_payload = {"projectId": str(PROJECT_ID)}
        if folder_id:
            filter_payload["folderId"] = str(folder_id)
            filter_payload["withChild"] = with_child
        
        initial_payload = {
            "filter": filter_payload,
            "startAt": 0,
            "maxResults": 1
        }
        initial_response = request_with_retry("POST", base_url, headers=get_headers(), json=initial_payload)
        total_reported = initial_response.json().get("total", 0)
        
        print(f"📊 API reports {total_reported} total test cases")
        
        # Fetch with larger page size to work around pagination issues
        while True:
            fetch_count = MAX_RESULTS
            if limit and (len(all_testcases) + MAX_RESULTS) > limit:
                fetch_count = limit - len(all_testcases)
            
            payload = {
                "filter": filter_payload,
                "startAt": start_at,
                "maxResults": fetch_count
            }
            
            response = request_with_retry(
                "POST", base_url, headers=get_headers(), json=payload
            )
            
            data = response.json()
            testcases = data.get("data", [])
            
            # Deduplicate by ID
            new_count = 0
            for tc in testcases:
                tc_id = tc.get("id")
                if tc_id and tc_id not in seen_ids:
                    seen_ids.add(tc_id)
                    all_testcases.append(tc)
                    new_count += 1
            
            display_total = min(total_reported, limit) if limit else total_reported
            current_count = min(len(all_testcases), limit) if limit else len(all_testcases)
            print(f"📥 Fetched {current_count} / {display_total} test cases (page returned {len(testcases)}, {new_count} new)")
            
            if limit and len(all_testcases) >= limit:
                all_testcases = all_testcases[:limit]
                break
            
            # If no new items were added, pagination is broken - stop
            if new_count == 0:
                if len(all_testcases) < total_reported:
                    print(f"⚠️  API pagination issue: only {len(all_testcases)} unique items returned out of {total_reported} reported")
                break
            
            if not testcases or len(all_testcases) >= total_reported:
                break
            
            start_at += fetch_count
            time.sleep(0.1)
        
        print(f"📥 Total: {len(all_testcases)} unique test cases")
    
    # Fetch details (including summary) and steps for each test case
    print(f"\n🔄 Fetching details and steps for {len(all_testcases)} test cases...")
    
    for i, tc in enumerate(all_testcases, 1):
        tc_id = tc.get("id")
        version_no = tc.get("version", {}).get("versionNo", 1)
        
        # Fetch test case details (summary, priority, status)
        try:
            url = f"{QMETRY_API_URL}/testcases/{tc_id}/versions/{version_no}"
            response = request_with_retry("GET", url, headers=get_headers())
            details = response.json().get("data", {})
            
            # Add summary and other details
            tc["summary"] = details.get("summary", "")
            tc["priority"] = details.get("priority", {})
            tc["status"] = details.get("status", {})
            
        except Exception as e:
            pass  # Skip if we can't get details
        
        # Fetch test steps using POST endpoint
        try:
            steps_url = f"{QMETRY_API_URL}/testcases/{tc_id}/versions/{version_no}/teststeps/search"
            steps_response = request_with_retry("POST", steps_url, headers=get_headers(), json={})
            steps_data = steps_response.json()
            
            steps = steps_data.get("data", [])
            
            # Extract stepDetails and expectedResults into separate objects
            step_details = {}
            expected_results = {}
            for step in steps:
                seq_no = step.get("seqNo", 0)
                step_details[f"step_{seq_no}"] = step.get("stepDetails", "")
                expected_results[f"step_{seq_no}"] = step.get("expectedResult", "")
            
            tc["stepDetails"] = step_details
            tc["expectedResults"] = expected_results
            tc["stepsCount"] = len(steps)
            
        except Exception as e:
            tc["stepDetails"] = {}
            tc["expectedResults"] = {}
            tc["stepsCount"] = 0
        
        if i % 20 == 0 or i == len(all_testcases):
            print(f"   ⚙️ Processed {i}/{len(all_testcases)} test cases")
        
        time.sleep(0.02)
    
    # Get parent folder info (id and name)
    parent_folder_info = get_folder_info(folder_id) if folder_id else {"id": None, "name": None}
    
    # Return result with project and parent folder info
    result = {
        "projectId": str(PROJECT_ID),
        "parentFolder": parent_folder_info,
        "testcases": all_testcases
    }
    
    return result


# ---------------- RUN ----------------
if __name__ == "__main__":
    import sys
    
    # Validate API key is set
    if not get_api_key():
        print("❌ Error: QMETRY_API_KEY environment variable is not set.")
        print("")
        print("To set it, choose one of these methods:")
        print("")
        print("  Option 1: Create a .env file in the project root:")
        print("    QMETRY_API_KEY=your-api-key-here")
        print("")
        print("  Option 2: Export as environment variable:")
        print("    export QMETRY_API_KEY='your-api-key-here'")
        print("")
        sys.exit(1)
    
    # Check for flags
    no_folder = "--no-folder" in sys.argv  # Use --no-folder to disable folder filter
    no_child = "--no-child" in sys.argv
    traverse_folders = "--traverse" in sys.argv  # Traverse each child folder individually
    
    # Parse folder ID if provided (--folder-id=123456), otherwise use QMETRY_FOLDER_ID from .env
    folder_id = get_folder_id()  # Get from QMETRY_FOLDER_ID in .env
    for arg in sys.argv[1:]:
        if arg.startswith("--folder-id="):
            folder_id = arg.split("=")[1]
    
    # Disable folder filter if --no-folder flag is set
    if no_folder:
        folder_id = None
    
    # Optional: pass max number of test cases as argument for testing
    max_tc = None
    skip_args = {"--no-folder", "--no-child", "--traverse"}
    for arg in sys.argv[1:]:
        if arg not in skip_args and not arg.startswith("--folder-id="):
            try:
                max_tc = int(arg)
                print(f"📝 Running with limit: {max_tc} test cases")
            except ValueError:
                pass
    
    # Print configuration
    print("🚀 Fetching test cases from QMetry API...")
    print(f"   Project ID: {PROJECT_ID}")
    if folder_id:
        print(f"   Folder ID: {folder_id}")
        if traverse_folders:
            print("   Mode: Traverse child folders and map test cases")
        else:
            print(f"   Include child folders: {not no_child}")
    if max_tc:
        print(f"   Limit: {max_tc}")
    print()
    
    # Fetch test cases
    result = fetch_test_cases(
        limit=max_tc, 
        folder_id=folder_id,
        with_child=not no_child,
        traverse_folders=traverse_folders
    )
    
    # Save results to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Print summary
    tc_count = len(result.get("testcases", []))
    folder_count = len(result.get("folders", []))
    if folder_count > 0:
        print(f"\n✅ Saved {tc_count} test cases from {folder_count} folders to {OUTPUT_FILE}")
    else:
        print(f"\n✅ Saved {tc_count} test cases to {OUTPUT_FILE}")

