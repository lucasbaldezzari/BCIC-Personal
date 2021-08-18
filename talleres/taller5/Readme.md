# Taller 5

![#3ab06f](https://via.placeholder.com/15/3ab06f/000000?text=+) `Comunicación entre PC y Arduino mediante Python para sincronizar estímulos y recepción/envío de información.  Comunicación entre Python y sitio en HTML para sincronizar estímulos. Manejo de versiones en archivos de firmware y hardware.`

## Requisitos

### Entorno de desarrollo para Arduino

Utilizaremos el entorno de Arduino para programar el microcontrolador y poder comunicarnos entre el micro y la pc a través del puerto serie.

#### IDE de desarrollo

Se recomienda utilizar un editor de texto o un IDE para trabajar. Algunos recomendados pueden ser,

- [Sublime Text](https://www.sublimetext.com/3): Es un editor de texto potente. Soporta la gran mayoría de los lenguajes existentes.
- [Notepad++](https://notepad-plus-plus.org/downloads/): Es un editor de texto potente. Soporta la gran mayoría de los lenguajes existentes.
- [Spyder](https://www.spyder-ide.org/): Es un entorno de desarrollo sumamente potente para trabajar con Python.
- [VisualStudio](https://code.visualstudio.com/) Entorno de desarrollo sumamente potente para trabajar no sólo con Python, sino también con otros lenguajes.

### Anaconda

Descargar e instalar "Anaconda" para la administración de ambientes de trabajo.

#### Instalar un Enviroment en Conda para trabajar durante el taller

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `IMPORTANTE: Peuden utilizar los mismos paquetes que los utilizados en el taller 4 pero agregando solamente la libreria Pyserial.`

Pasos propuestos a seguir:

- _Abrir la consola de Anaconda, de Windows o Linux._
- _Ejecutar:_ conda install --name base nb_conda_kernels
- _Moverse hasta el directorio donde se almacenará el trabajo_
- _Ejecutar:_ conda env update --file dependencias.yml (Nota: el nombre por defecto del enviroment es "taller5", para cambiarlo debe editarse el archivo dependencias.yml)
- _Activar el ambiente:_ conda activate taller4-bcic

Al finalizar el proceso deberían ver un mensaje similar a este:

_To activate this environment, use_

     $ conda activate taller5

_To deactivate an active environment, use_

     $ conda deactivate

### Dependencias

Paquetes necesarios.

- python3.8
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