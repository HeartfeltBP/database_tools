<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/HeartfeltBP/database_tools/README">
    <img src="images/heartfelt-logo.png" alt="Heartfelt Logo" width="400" height="400">
  </a>

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
        <li><a href="#generate-records">Generate Records</a></li>
        <li><a href="#read-records">Read Records</a></li>
      </ul>
    <li><a href="#signal-processing-pipeline">Signal Processing Pipeline</a></li>
    <li><a href="#database-structure">Database Structure</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- Introduction -->
## Introduction

This repo contains a set of tools for extracting and cleaning photoplethsmothography (ppg) and artial blood pressure (abp) waveforms from the [MIMIC-III Waveforms Database](https://physionet.org/content/mimic3wdb/1.0/) for the purpose of blood pressure estimation via deep learning. 

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
The functions in this library rely on a data folder named with the convention `data-YYYY-MM-DD`. This directory contains two additional folders, `mimic3` and `figures`. The `mimic3/lines` folder is intended to hold the jsonlines files the data will initially saved to. The `mimic3/records` folder will hold the TFRecords files generated from these jsonlines files. This will be discussed in greater depth in the <a href="#generate-records">Generate Records</a> section.

### Get Valid Records
The class DataLocator is specifically written to find valid data files in the MIMIC-III Waveforms subset and create a csv of the html links for these data files. Performing this task prior to downloading is done to improve runtime and the usability of this workflow. Valid records refers to data files that contain both PPG and ABP recordings and are at least 10 minutes in length. Currently this code is only intended for the MIMIC-III Waveforms subset but will likely be adapated to allow for valid segments to be identified in the MIMIC-III Matched Subset (records are linked to clinical data). To perform an extraction the file `scripts/get-valid-segs.py` can be run (data directory and repository path must be configured manually). This function will output a csv called `valid-segments.csv` to the data directory provided.

### Build Database

### Generate Records

### Read Records

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- SIGNAL PROCESSING PIPELINE -->
## Signal Processing Pipeline

Placeholder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- DATABASE STRUCTURE -->
## Database Structure

Placeholder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
