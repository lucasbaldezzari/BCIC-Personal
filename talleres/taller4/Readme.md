# Taller 4 - Primer competencia de vehículos robóticos controlados por interfaces cerebro computadora

## Requisitos

### Anaconda

Descargar e instalar "Anaconda" para la administración de ambientes de trabajo.

#### IDE de desarrollo

Se recomienda utilizar un editor de texto o un IDE para trabajar. Algunos recomendados pueden ser,

- [Sublime Text](https://www.sublimetext.com/3): Es un editor de texto potente. Soporta la gran mayoría de los lenguajes existentes.
- [Notepad++](https://notepad-plus-plus.org/downloads/): Es un editor de texto potente. Soporta la gran mayoría de los lenguajes existentes.
- [Spyder](https://www.spyder-ide.org/): Es un entorno de desarrollo sumamente potente para trabajar con Python.

#### Instalar un Enviroment en Conda para trabajar durante el taller

Pasos propuestos a seguir:

- _Abrir la consola de Anaconda, de Windows o Linux._
- _Ejecutar:_ conda install --name base nb_conda_kernels
- _Moverse hasta el directorio donde se almacenará el trabajo_
- _Ejecutar:_ conda env update --file dependencias.yml (Nota: el nombre por defecto del enviroment es "taller4-BCIC", para cambiarlo debe editarse el archivo dependencias.yml)
- _Activar el ambiente:_ conda activate taller4-BCIC

Al finalizar el proceso deberían ver un mensaje similar a este:

_To activate this environment, use_

     $ conda activate taller2-BCIC

_To deactivate an active environment, use_

     $ conda deactivate

### Dependencias

Paquetes necesarios.

- python>=3.7
- matplotlib
- pip
- brainflow
- pyqtgraph
- numpy
- pandas
- scikit-learn
- scipy
- keyboard