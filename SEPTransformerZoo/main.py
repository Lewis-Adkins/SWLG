# from core.forecasting_model_run import ModelRunSetUp
from core.setup import SetUp
from core.model_run import ModelRun

from pathlib import Path




def main():
    setup = SetUp()
    setup.check_boostrap_data_exists()


    
    for pt in setup.config["data"]["prediction_times"]:
        
            
            


if __name__ == "__main__":
    main()