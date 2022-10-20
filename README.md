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

Placeholder.

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
