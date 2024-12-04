# Exemplar-based Inpainting

Python implementation of the exemplar-based inpainting method of Criminisi et al.:

> `Criminisi A, Pérez P, Toyama K. Region filling and object removal by exemplar-based image inpainting[J]. IEEE Transactions on image processing, 2004, 13(9): 1200-1212.`

## Installation

This project requires Python >= 3.7. To install it using `pip`:

```
cd <path_to>/exemplar_based_inpainting
pip install .
```

## Compile the docs

This project uses `mkdocs`. Therefore, compiling the documentation is as simple as running the following command from the same directory containing the `mkdocs.yml` file:

```
mkdocs build
```

And serving the documentation to read it locally:

```
mkdocs serve
```

## Usage

After installation, you should have the `exemplar_based_inpainting` command line tool available.

The only required parameter is the input image to inpaint. If you want to manually set the inpainting mask, and you do not need to store the results, you can just call it as follows:

```
exemplar_based_inpainting <image_path>
```

Please check the documentation for more information on the different parameters of this tool. We also provide some examples to test the tool in the `data` folder of this project.

## Acknowledgements

This project has been developed by Coronis Computing S.L. within the EMODnet Bathymetry (High Resolution Seabed Mapping) project.

* EMODnet: http://www.emodnet.eu/
* EMODnet (bathymetry): http://www.emodnet-bathymetry.eu/
* Coronis: http://www.coronis.es