# Optimal Waveforms for Dipole Moment Estimation with Coherent States:
This code can be used to reproduce results in https://arxiv.org/abs/2509.XXXXX. <br>

## Code Organization:
The repository is organized into five folders:

1. __`Figure2_3_4_standard_pulses`__  
    Contains Python code for computing the quantum Fisher information (QFI) of all standard pulses. 
    This code can be used to reproduce Figures 3, 4, and 6. It also includes routines for optimizing 
    the pulse width as a function of the average photon number.

2. __`Figure4_optimization_real_harmonics`__  
    Provides Python code for optimizing the QFI of an arbitrary real pulse expressed in a harmonic basis, 
    for a given width, as a function of the average photon number. This code can be used to reproduce Figure 5.

3. __`Figure5_optimization_real_hermite_gaussian`__  
    Extends the above optimization to the Hermite-Gaussian basis, ensuring that no optimal pulse is missed.
   This code can be used to reproduce Figure 7.

5. __`Optimization_complex_pulses`__  
    Contains code for optimization in the complex harmonics basis.

6. __`Long_width_limit`__  
    Provides code that compares the QFI of a coherent state in the long-width limit with that of 
    the single-photon pulse.


## Citing:
If you find this code useful in your research, please consider citing our paper:

```bib
@article{chinni2025optimal,
  title={Optimal Waveforms for Dipole Moment Estimation with Coherent States},
  author={Chinni, Karthik and Quesada, Nicol{\'a}s},
  journal={arXiv preprint arXiv:2509.XXXXX},
  year={2025}
}
```

## Funding:
Funding for our work has been provided by
* Ministère de l'Économie et de l’Innovation du Québec, 
* Natural Sciences and Engineering Research Council of Canada,
* Fonds de recherche du Québec-Nature et technologies (FRQNT) under the Programme PBEEE / Bourses de stage postdoctoral
