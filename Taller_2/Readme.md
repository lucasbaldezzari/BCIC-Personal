# Requisitos para el taller

## Anaconda

Descargar e instalar "Anaconda" para la administración de ambientes de trabajo.

### IDE de desarrollo

Se recomienda utilizar un editor de texto o un IDE para trabajar. Algunos recomendados pueden ser,

- [Sublime Text](https://www.sublimetext.com/3): Es un editor de texto potente. Soporta la gran mayoría de los lenguajes existentes.
- [Spyder](https://www.spyder-ide.org/): Es un entorno de desarrollo sumamente potente para trabajar con Python.

### Instalar un Enviroment en Conda para trabajar durante el taller

Pasos a seguir:

- _Abrir la consola de Anaconda, de Windows o Linux._
- _Ejecutar:_ conda install --name base nb_conda_kernels
- _Moverse hasta el directorio donde se almacenará el trabajo_
- _Ejecutar:_ conda env update --file dependencias.yml (Nota: el nombre por defecto del enviroment es "taller2-BCIC", para cambiarlo debe editarse el archivo dependencias.yml)
- _Activar el ambiente:_ conda activate taller2-BCIC

Al finalizar el proceso deberían ver un mensaje similar a este:

_To activate this environment, use_

     $ conda activate taller2-BCIC

_To deactivate an active environment, use_

     $ conda deactivate

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