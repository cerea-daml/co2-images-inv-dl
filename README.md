# co2-images-inv-pp



Welcome to the official repository for the paper "Deep learning applied to CO2 power plant emissions quantification using simulated satellite images," submitted to the "Geoscientific Model Development" journal (COPERNICUS).


## Project Overview

This project demonstrates a novel concept that deep learning can be employed to identify anthropogenic XCO2 emissions from hotspots in XCO2 images, assisted by NO2 and wind data.

Our scripts and modules are written in Python, using Tensorflow as the deep learning framework.

To employ these scripts, download the datasets of fields and plumes from [inv-zenodo](https://doi.org/10.5281/zenodo.8095487).
Alternatively, you can create the netcdf datasets directly from the [SMARTCARB](https://zenodo.org/record/4034266#.Yt6btp5BzmE) dataset. 
Note that the data generation scripts are not part of this repository, but can be provided upon request.

Weights for pre-trained models can be obtained from [inv-zenodo-weights](https://doi.org/10.5281/zenodo.8095487).

Within the 'examples' directory, you will find two Jupyter notebooks, `train.ipynb` and `test.ipynb`:
- `train.ipynb` guides on how to train a model using a configuration (cfg) file, configured with Hydra.
- `test.ipynb` outlines how to evaluate a pre-trained model.
For both files, specific items in the configuration file that need adjustments for your set up are highlighted.

Post data collection/generation, use our `main.py` Python script (refer `examples/train.ipynb`) to train the Convolutional Neural Network as elaborated in the manuscript.

For any further queries, do not hesitate to reach out or create a GitHub issue.

## Acknowledgements and Authors

This project is funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement 958927 (Prototype system for a Copernicus CO2 service). 
CEREA is a proud member of Institut Pierre Simon Laplace (IPSL).

## Support

Feel free to contact: joffrey.dumont@enpc.fr
