# Requisitos para el taller

## Anaconda

Descargar e instalar "Anaconda" para la administración de ambientes de trabajo.

### Instalar un Enviroment en Conda para trabajar durante el taller

Pasos a seguir:

- Abrir la consola de Anaconda, de Windows o Linux.
- Ejecutar: conda install --name base nb_conda_kernels
- Moverse hasta el directorio donde se almacenará el trabajo
- Ejecutar: conda env update --file dependencias.yml (Nota: el nombre por defecto del enviroment es "teller2-BCIC", para cambiarlo debe editarse el archivo dependencias.yml)

## Dependencias

Deben descargar lo siguiente

- python>=3.7
- matplotlib
- pip
- mne
- h5py>=2.10.0
- Keras>=2.4.3
- Keras-Preprocessing>=1.1.2
- numpy>=1.18.5
- pandas>=1.1.3
- scikit-learn>=0.23.2
- scipy>=1.4.1
- tensorboard>=2.3.0
- tensorboard-plugin-wit>=1.7.0
- tensorflow>=2.3.1
- tensorflow-estimator>=2.3.0