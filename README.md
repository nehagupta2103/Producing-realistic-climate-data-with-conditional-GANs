## Installation

 - Clone the repository in your working folder
 - Create a virtual environment and install the `requirements.txt` file
   - `python -m venv env_folder`
   - `source env_folder/bin/activate`
   - `pip install -r requirements.txt`
- /!\ The code is in version tensorflow==1.14.0 and keras==2.2.4 as a consequence the virtual environment needs to be created using python 3.7.X .
- /!\ Cartopy package requires some preinstallation (see [documentation](https://scitools.org.uk/cartopy/docs/latest/installing.html))
- Download the data (a sample is available [here](https://www.kaggle.com/datasets/camilleb469/climate-data-3years-simulation)) and place it in the ./data/raw/ folder.
- Finally run `python training.py` .