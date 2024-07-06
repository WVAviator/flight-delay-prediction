# Flight Delay Probability Analysis and Prediction

This project aims to identify some common patterns surrounding flight delays using flight operations data to identify and visualize trends and patterns, assess components for data prediction, and develop a predictive model that can benefit businesses and individuals when planning flights and travel.

## Initial Setup

To review the Notebooks in this project, start by activating the virtual environment:

MacOS/Linux:
```shell
source venv/bin/activate
```

Windows:
```shell
venv\Scripts\activate
```

Run the following command in the project root to install all the required dependencies:

```shell
pip install -r requirements.txt
```

### Jupyter Notebook Files

To review `.ipynb` files, first install Jupyter Notebook:

```shell
pip install notebook
```

Then run Jupyter Notebook and navigate to this project's files.

```shell
jupyter notebook
```

Read and execute each cell of the notebooks in order, or simply select `Kernel > Restart & Run All` to run all cells in a Notebook.

## Project Notebooks

There are three notebooks to review in this project. 

Start by reviewing and executing the code in `data_preparation.ipynb`. This notebook discusses the wrangling, sampling, and cleaning of the data for this project. It includes instructions for downloading and resampling the full dataset if desired, although a preconfigured sample is included with the project files.

Next, review `visualizations.ipynb` to see the descriptive analysis of the data, which shows known correlations between independent and dependent variables and displays visualizations to understand those relationships.

Last, review `predictive_analysis.ipynb` to review the process of constructing, testing, and verifying a predictive model for flight delay prediction.

## User Interface

To use the model for flight delay probability predictions, run the following from a command line:

```shell
python main.py
```

This will start the program, create the model if it is not already pickled (saved), and then ask a series of questions to obtain flight information before predicting the probability of various delay severity categories.