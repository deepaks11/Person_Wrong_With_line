import os
import shutil


class Folder:

    def __init__(self, path):

        self.folder_path = path

    def create(self):
        try:

            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
        except Exception as error:
            print(error)

    def remove(self):
        try:
            if os.path.exists(self.folder_path):
                shutil.rmtree(self.folder_path, ignore_errors=True)
        except Exception as error:
            print(error)