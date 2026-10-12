from pathlib import Path
# import yaml
import numpy as np
import pandas as pd
import sys
class FileManager:
    def __init__(self):
        # self.config = self.get_config()
        pass


    # def get_config(self)->dict:

    #     with open("config.yaml", "r") as f:
    #         try:
    #             config = yaml.safe_load(f)
    #         except yaml.YAMLError as exc:
    #             print(f"[ERROR] Reading config.yaml failed: {exc}")


    def check_data_files(self, data_files: list[Path])->None:

        data_folder_path = Path("data")

        if not data_folder_path.is_dir():
            print(f"[ERROR] 'data' folder does not exist")
            sys.exit()

        data_files = [df.name for df in  data_folder_path.iterdir()]

        file_list_check = ["event_timestamps.txt, data.csv, event_indices.txt"]

        for flc in file_list_check:
            if not flc in data_files:
                print(f"[ERROR]{flc} not found in data folder")
                sys.exit()
        

    def get_data(self)-> pd.DataFrame:
        self.check_data_files()

        




            
fm = FileManager()
fm.get_data()



