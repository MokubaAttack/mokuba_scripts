import os
import dropbox

def up_drop(path,path2,url):
	dbx=dropbox.Dropbox(
		oauth2_refresh_token=url[2],
		app_key=url[0],
		app_secret=url[1]
		)
	f=open(path, "rb")
	dbx.files_upload(f.read(), "/"+path2,mode=dropbox.files.WriteMode("overwrite"))
	f.close()
	del dbx
	
def down_drop(path,path2,url):
	dbx=dropbox.Dropbox(
		oauth2_refresh_token=url[2],
		app_key=url[0],
		app_secret=url[1]
		)
	if os.path.exists(path2):
		os.remove(path2)
	dbx.files_download_to_file(path2, '/'+path)
	del dbx