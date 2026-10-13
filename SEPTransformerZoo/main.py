# from core.forecasting_model_run import ModelRunSetUp
from core.setup import SetUp
from pathlib import Path




def main():
    setup = SetUp()
    setup.check_boostrap_data_exists()



if __name__ == "__main__":
    main()