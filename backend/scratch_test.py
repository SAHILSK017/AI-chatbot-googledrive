from backend.google_drive import get_drive_service
import os

def get_all_subfolders(root_id):
    service = get_drive_service()
    folder_ids = [root_id]
    queue = [root_id]
    
    while queue:
        current_id = queue.pop(0)
        q = f"'{current_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        res = service.files().list(q=q, fields="files(id, name)").execute()
        for f in res.get('files', []):
            folder_ids.append(f['id'])
            queue.append(f['id'])
            
    return folder_ids

print("Subfolders:", get_all_subfolders(os.getenv("TARGET_FOLDER_ID")))
