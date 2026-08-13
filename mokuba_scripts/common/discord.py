import os
import shutil
import dropbox
import time

def to_discord(path,url):
	dbx=dropbox.Dropbox(
		oauth2_refresh_token=url[2],
		app_key=url[0],
		app_secret=url[1]
		)
	if os.path.isdir(path):
		shutil.make_archive('archive_shutil', format='zip', root_dir=path)
		ut=round(time.time())
		f=open('archive_shutil.zip',"rb")
		dbx.files_upload(f.read(), "/"+str(ut)+".zip")
		f.close()
		os.remove('archive_shutil.zip')
	else:
		f=open(path, "rb")
		dbx.files_upload(f.read(), "/"+os.path.basename(path))
		f.close()
