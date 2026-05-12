from backend.google_drive import search_drive
res = search_drive("trashed = false", page_size=10)
for r in res:
    print(r['name'], r['mimeType'])
