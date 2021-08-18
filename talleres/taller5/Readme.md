# Taller 5

Comunicación entre PC y Arduino mediante Python para sincronizar estímulos y recepción/envío de información.  Comunicación entre Python y sitio en HTML para sincronizar estímulos. Protocolo de adquisición de señales de EEG para entrenar clasificadores que serán utilizados en la BCI de manera online. Manejo de versiones en archivos de firmware y hardware.

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

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `IMPORTANTE: Los paquetes son los mismos que los utilizados en el taller 4 pero agregando solamente la libreria Pyserial. De esta manera podrían usar el mismo enviroment del taller4 pero instalando la libreria mencionada. No obstante, pueden crear un nuevo enviroment si lo desean.`

Pasos propuestos a seguir:

- _Abrir la consola de Anaconda, de Windows o Linux._
- _Ejecutar:_ conda install --name base nb_conda_kernels
- _Moverse hasta el directorio donde se almacenará el trabajo_
- _Ejecutar:_ conda env update --file dependencias.yml (Nota: el nombre por defecto del enviroment es "taller4-BCIC", para cambiarlo debe editarse el archivo dependencias.yml)
- _Activar el ambiente:_ conda activate taller4-bcic

Al finalizar el proceso deberían ver un mensaje similar a este:

_To activate this environment, use_

     $ conda activate taller5-bcic

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