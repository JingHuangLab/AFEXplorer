# AFEXplorer

AFEXplorer (AFEX) is a generic approach that tailors AlphaFold predictions to user-defined constraints in coarse coordinate spaces by optimizing embedding features.
It has been demonstrated to be effective in predicting alternative structures of proteins conditioned by local or global structure features for human kinases and membrane transporters.


## Installation

Assume you have a working AlphaFold 2.3 environment, and install the following packages depending on your JAX/JAXLIB version in the environment. For more info, see the corresponding GitHub repos and find a compatible release.
```bash
chex
optax
```

### Kincore-standalone is also required when predicting human kinases.

Simply, `hmmer` needs to be installed in the current environment to support Kincore. See [Kincore-standalone](https://github.com/vivekmodi/Kincore-standalone) for more info.

## Use AFEX for alternative structure prediction

1. Prepare input.

    Run AlphaFold for `features.pkl`, which is the input of AFEX.

2. Take AdK as an example.
   * Add corresponding CV loss for the open state in `afexplore_optim.py`.
   * Set the directory of AF model parameters as $AF_PARAM.
   * Run AFEX.

        ```bash
        cd scripts
        bash run_afexplore_optim.sh ../data_afexplore_monomer_ADKopen ADK out $AF_PARAM 
        ```
3. The output put will be in `data_afexplore_monomer_ADKopen/ADK`.
4. It is flexible for user to adjust the AFEX optimizaiton-related parameters (e.g. learning rate) in `afexplore_optim.py` and `run_afexplore_optim.sh`.

Note AFEX currently is compatible solely with AF-Monomer.

## Reference
Tengyu Xie, Zilin Song, Jing Huang. Conditioned Protein Structure Prediction. bioRxiv 2023.09.24.559171; doi: https://doi.org/10.1101/2023.09.24.559171.

## Acknowledgements

Inspiration, code snippets, etc.

* [AlphaFold](https://github.com/google-deepmind/alphafold)
* [AFProfile](https://github.com/patrickbryant1/AFProfile)
* [Kincore-standalone](https://github.com/vivekmodi/Kincore-standalone)

