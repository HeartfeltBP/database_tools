<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
      
            
 <img src="images/heartfelt-logo.png"  width="300em" height="300em"> 

  
  <h1 align="center">MIMIC-III Database Tools</h1>

  <p align="center">
    For extracting and cleaning ppg and abp data from the MIMIC-III Waveforms Database.
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
        <li><a href="#poetry">Poetry</a></li>
        <li><a href="#get-valid-records">Get Valid Records</a></li>
        <li><a href="#build-database">Build Database</a></li>
        <li><a href="#evaluate-dataset">Evaluate Dataset</a></li>
        <li><a href="#generate-records">Generate Records</a></li>
        <li><a href="#read-records">Read Records</a></li>
      </ul>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- Introduction -->
## Introduction

This repo contains a set of tools for extracting and cleaning photoplethysmography (ppg) and artial blood pressure (abp) waveforms from the [MIMIC-III Waveforms Database](https://physionet.org/content/mimic3wdb/1.0/) for the purpose of blood pressure estimation via deep learning. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This sections details the requirements to start using this library. Links are for Ubuntu installation.

### Prerequisites

1. Python
```shell
sudo apt install python3.8 -y
sudo apt install python3.8-dev python3.8-venv -y

echo 'export PATH="$PATH:/home/ubuntu/.local/bin"' >> ~/.bashrc
source ~/.bashrc

curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8
python3.8 -m pip install virtualenv
python3.8 -m venv .venv/base-env
echo 'alias base-env="source ~/.venv/base-env/bin/activate"' >> ~/.bashrc
base-env

python3.8 -m pip install --upgrade pip
```
2. Poetry
```shell
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
poetry --version
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Poetry
The commands below can be used to install the poetry environment, build the project, and activate the environment.
```shell
cd database-tools
poetry lock
poetry install
poetry build
poetry shell
```

### Create Data Directory
The functions in this library rely on a data folder named with the convention `data-YYYY-MM-DD`. This directory contains two additional folders, `mimic3/` and `figures/`. The `mimic3/lines/` folder is intended to hold the jsonlines files the data will initially saved to. The `mimic3/records/` folder will hold the TFRecords files generated from these jsonlines files. This will be discussed in greater depth in the <a href="#generate-records">Generate Records</a> section.

### Get Valid Records
The class DataLocator (located in `database_tools/tools/`) is specifically written to find valid data files in the MIMIC-III Waveforms subset and create a csv of the html links for these data files. Performing this task prior to downloading is done to improve runtime and the usability of this workflow. Valid records refers to data files that contain both PPG and ABP recordings and are at least 10 minutes in length. Currently this code is only intended for the MIMIC-III Waveforms subset but will likely be adapated to allow for valid segments to be identified in the MIMIC-III Matched Subset (records are linked to clinical data). To perform an extraction the file `scripts/get-valid-segs.py` can be run (data directory and repository path must be configured manually). This function will output a csv called `valid-segments.csv` to the data directory provided. The figure below shows how these signals are located.

Add mimic3 valid segs logic figure.

### Build Database
The class `BuildDatabase` (located in `database_tools/tools/`) downloads data from `valid-segments.csv`, extracts PPG and ABP data, and then processed it by leveraging the `SignalProcessor` class (located in `database_tools/preprocessing/`). A database can be build by running `scripts/build_database.py` (be sure to configure the paths). BuildDatabase takes a few important parameters which modify how signals are excluded and how the signals are divided prior to processing. The `win_len` parameter controls the length of each window, `fs` is the sampling rate of the data (125 Hz in the case of MIMIC-III), while `samples_per_file`, `samples_per_patient`, and `max_samples` control the size of the dataset (how many files the data is spread across, how many samples a patient can contribute, and the total number of samples in the dataset. The final parameter `config` controls the various constants of the SignalProcessor that determine the quality threshold for accepting signals. The SignalProcessor filters signals according to the figure chart below. The functions used for this filtering can be found in `database_tools/preprocessing/`. Data exctracted with this script is saved directly to the `mimic3/lines/` folder in the data directory. A file named `mimic3_stats.csv` containing the stats of every processed waveform (not just the valid ones) will also be saved to the data directory.

Add data preprocessing figure.

### Evaluate Dataset
The class `DataEvaluator` (located in `database_tools/tools/`) reads the `mimic3_stats.csv` file from the provided data directory and outputs figures to visualize the statistics. These figures are saved directly to the `figures/` folder in the data directory in addition to be output such that they can be viewed in a Jupyter notebook. The 3D histogram are generated using the fuction `histogram3d` located in `database_tools/plotting/`.

### Generate Records
Once data has been extracted TFRecords can be generated for training a Tensorflow model. The class `RecordsHandler` contains the method `GenerateRecords` which is used to create the TFRecords. This can be done using `scripts/generate_records.py` (paths must be configured). When calling `GenerateRecords` the size of the train, validation, and test splits, as well as the max number of samples per file and a boolean to control whether or not the data is standardized must be specified (using `sklearn.preprocessing.StandardScaler()`.

### Read Records
The class `RecordsHandler` also contains the function `ReadRecords` which can be used to read the TFRecords into a Tensorflow `TFRecordsDataset` object. This function can be used to inspect the integrity of the dataset or for loading the dataset for model training. The number of cores and a TensorFlow `AUTOTUNE` object must be provided.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
