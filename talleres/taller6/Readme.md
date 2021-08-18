# Taller 6

Taller dedicado para aprender adquisición y procsamiento de EEG -a través de la placa Cyton de OpenBCI- en tiempo real usando Python. Protocolo de adquisición de señales de EEG para entrenar diferentes clasificadores de SSVEPs, comparar sus performances y utilizarlos para clasificar  de manera offline. 

![#3ab06f](https://via.placeholder.com/15/3ab06f/000000?text=+) `Estos clasificadores serán utilizados en la BCI de manera online para detectar comandos y controlar el vehículo robótico.`

## Requisitos

### Anaconda

Descargar e instalar "Anaconda" para la administración de ambientes de trabajo.

#### IDE de desarrollo

Se recomienda utilizar un editor de texto o un IDE para trabajar. Algunos recomendados pueden ser,

- [Sublime Text](https://www.sublimetext.com/3): Es un editor de texto potente. Soporta la gran mayoría de los lenguajes existentes.
- [Notepad++](https://notepad-plus-plus.org/downloads/): Es un editor de texto potente. Soporta la gran mayoría de los lenguajes existentes.
- [Spyder](https://www.spyder-ide.org/): Es un entorno de desarrollo sumamente potente para trabajar con Python.
- [VisualStudio](https://code.visualstudio.com/) Entorno de desarrollo sumamente potente para trabajar no sólo con Python, sino también con otros lenguajes.

#### Instalar un Enviroment en Conda para trabajar durante el taller

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `IMPORTANTE: Los paquetes para este taller son los mismos que los utilizados en el taller 5.`

Pasos propuestos a seguir:

- _Abrir la consola de Anaconda, de Windows o Linux._
- _Ejecutar:_ conda install --name base nb_conda_kernels
- _Moverse hasta el directorio donde se almacenará el trabajo_
- _Ejecutar:_ conda env update --file dependencias.yml (Nota: el nombre por defecto del enviroment es "taller6", para cambiarlo debe editarse el archivo dependencias.yml)
- _Activar el ambiente:_ conda activate taller4-bcic

Al finalizar el proceso deberían ver un mensaje similar a este:

_To activate this environment, use_

     $ conda activate taller6

_To deactivate an active environment, use_

     $ conda deactivate

### Dependencias

Paquetes necesarios.

- python3.8 (o superior)
- matplotlib
- pip
- brainflow
- pyqtgraph
- numpy
- pandas
- scikit-learn
- scipy
- keyboard
- Pyserial