"""SVMTrainingModule V2.0"""


import os
import numpy as np
import numpy.matlib as npm
import json

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


from scipy.signal import butter, filtfilt, windows
from scipy.signal import welch

import pickle

import matplotlib.pyplot as plt

from utils import filterEEG
import fileAdmin as fa


class SVMTrainingModule():

    def __init__(self, rawDATA, PRE_PROCES_PARAMS, FFT_PARAMS, frecStimulus, nchannels,nsamples,ntrials,modelName = ""):
        """Variables de configuración

        Args:
            - rawDATA(matrix[clases, samples, trials]): Señal de EEG
            - subject (string o int): Número de sujeto o nombre de sujeto
            - PRE_PROCES_PARAMS: Parámetros para preprocesar los datos de EEG
            - FFT_PARAMS: Parametros para computar la FFT
            - modelName: Nombre del modelo
        """

        self.rawDATA = rawDATA

        if not modelName:
            self.modelName = f"SVMModel"

        else:
            self.modelName = modelName

        self.frecStimulus = frecStimulus
        self.nclases = len(frecStimulus)
        self.nchannels = nchannels
        self.nsamples = nsamples
        self.ntrials = ntrials

        self.dataBanked = None #datos de EEG filtrados con el banco
        self.model = None
        self.trainingData = None
        self.labels = None

        self.signalPSD = None #PSD de mis datos
        self.signalSampleFrec = None

        # self.lda = LinearDiscriminantAnalysis(n_components = self.nclases - 1)

        self.MSF = np.array([]) #Magnitud Spectrum Features

        self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
        self.FFT_PARAMS = FFT_PARAMS

        self.METRICAS = {f'modelo_{self.modelName}': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                                                      'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}}}

    def createSVM(self, kernel = "rbf", gamma = "scale", C = 1, probability=False, randomSeed = None):
        """Se crea modelo"""

        self.model = SVC(C = C, kernel = kernel, gamma = gamma, probability=probability, random_state = randomSeed)

        return self.model

    def applyFilterBank(self, eeg, bw = 2.0, order = 4):
        """Aplicamos banco de filtro a nuestros datos.
        Se recomienda aplicar un notch en los 50Hz y un pasabanda en las frecuencias deseadas antes
        de applyFilterBank()
        
        Args:
            - eeg: datos a aplicar el filtro. Forma [clase, samples, trials]
            - frecStimulus: lista con la frecuencia central de cada estímulo/clase
            - bw: ancho de banda desde la frecuencia central de cada estímulo/clase. Default = 2.0
            - order: orden del filtro. Default = 4"""

        nyquist = 0.5 * self.FFT_PARAMS["sampling_rate"]
        signalFilteredbyBank = np.zeros((self.nclases,self.nsamples,self.ntrials))
        for clase, frecuencia in enumerate(self.frecStimulus):   
            low = (frecuencia-bw/2)/nyquist
            high = (frecuencia+bw/2)/nyquist
            b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
            signalFilteredbyBank[clase] = filtfilt(b, a, eeg[clase], axis = 0) #filtramos

        self.dataBanked = signalFilteredbyBank
        return self.dataBanked

    def computWelchPSD(self, signalBanked, fm, ventana, anchoVentana, average = "median", axis = 1):
        """Computa la transformada de Welch al EEG"""
        self.signalSampleFrec, self.signalPSD = welch(signalBanked, fs = fm, window = ventana, nperseg = anchoVentana, average='median',axis = axis)
        return self.signalSampleFrec, self.signalPSD


    def featuresExtraction(self, ventana, anchoVentana = 5, bw = 2.0, order = 4, axis = 1):
        """EXtracción de características a partir de datos de EEG sin procesar"""
        filteredEEG = filterEEG(self.rawDATA, self.PRE_PROCES_PARAMS["lfrec"],
                                self.PRE_PROCES_PARAMS["hfrec"],
                                self.PRE_PROCES_PARAMS["order"],
                                self.PRE_PROCES_PARAMS["bandStop"],
                                self.PRE_PROCES_PARAMS["sampling_rate"],
                                axis = axis)

        dataBanked = self.applyFilterBank(filteredEEG, bw=bw, order = 4) #Aplicamos banco de filtro

        anchoVentana = int(self.PRE_PROCES_PARAMS["sampling_rate"]*anchoVentana) #fm * segundos
        ventana = ventana(anchoVentana)

        self.signalSampleFrec, self.signalPSD = self.computWelchPSD(dataBanked,
                                                fm = self.PRE_PROCES_PARAMS["sampling_rate"],
                                                ventana = ventana, anchoVentana = anchoVentana,
                                                average = "median", axis = axis)

        return self.signalSampleFrec, self.signalPSD

    #Transforming data for training
    def getDataForTraining(self, features):
        """Preparación del set de entrenamiento.

        Argumentos:
            - features: Parte Real del Espectro or Parte Real e Imaginaria del Espectro
            con forma [número de características x canales x clases x trials x número de segmentos]
            - clases: Lista con las clases para formar las labels

        Retorna:
            - trainingData: Set de datos de entrenamiento para alimentar el modelo SVM
            Con forma [trials*clases x number of features]
            - Labels: labels para entrenar el modelo a partir de las clases
        """

        numFeatures = features.shape[1]
        trainingData = features.swapaxes(2,1).reshape(self.nclases*self.ntrials, numFeatures)

        classLabels = np.arange(self.nclases)

        labels = (npm.repmat(classLabels, self.ntrials, 1).T).ravel()

        return trainingData, labels

    def trainAndValidateSVM(self, clases, test_size = 0.2, randomSeed = None, applyLDA = False):
        """Método para entrenar un modelo SVM.

        Argumentos:
            - clases (int): Lista con valores representando la cantidad de clases
            - test_size: Tamaño del set de validación"""

        self.frecStimulus = clases

        self.trainingData, self.labels = self.getDataForTraining(self.signalPSD)

        X_trn, X_val, y_trn, y_val = train_test_split(self.trainingData, self.labels, test_size = test_size, shuffle = True, random_state = randomSeed)

        # if applyLDA == True:
        #     self.lda.fit(X_trn, y_trn)
        #     X_trn = self.lda.transform(X_trn)
        #     X_val = self.lda.transform(X_val)

        self.model.fit(X_trn, y_trn)

        y_pred = self.model.predict(X_trn)

        precision, recall, f1,_ = precision_recall_fscore_support(y_trn, y_pred, average='weighted')

        self.METRICAS = {f'modelo_{self.modelName}': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                                                      'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}}}

        precision, recall, f1,_ = precision_recall_fscore_support(y_trn, y_pred, average='macro')
        accuracy = accuracy_score(y_trn, y_pred)


        self.METRICAS[f'modelo_{self.modelName}']['trn']['Pr'] = precision
        self.METRICAS[f'modelo_{self.modelName}']['trn']['Rc'] = recall
        self.METRICAS[f'modelo_{self.modelName}']['trn']['Acc'] = accuracy
        self.METRICAS[f'modelo_{self.modelName}']['trn']['F1'] = f1

        y_pred = self.model.predict(X_val)

        precision, recall, f1,_ = precision_recall_fscore_support(y_val, y_pred, average='macro')
        accuracy = accuracy_score(y_val, y_pred)

        self.METRICAS[f'modelo_{self.modelName}']['val']['Pr'] = precision
        self.METRICAS[f'modelo_{self.modelName}']['val']['Rc'] = recall
        self.METRICAS[f'modelo_{self.modelName}']['val']['Acc'] = accuracy
        self.METRICAS[f'modelo_{self.modelName}']['val']['F1'] = f1

        return self.METRICAS

    def saveTrainingSignalPSD(self, signalPSD, path, filename = ""):

        actualFolder = os.getcwd()#directorio donde estamos actualmente
        os.chdir(path)
        
        if not filename:
            filename = self.modelName

        np.savetxt(f'{filename}_signalPSD.txt', signalPSD, delimiter=',')
        np.savetxt(f'{filename}_signalSampleFrec.txt', self.signalSampleFrec, delimiter=',')

        os.chdir(actualFolder)

    def saveModel(self, path, filename = ""):
        """Método para guardar el modelo"""

        actualFolder = os.getcwd()#directorio donde estamos actualmente
        os.chdir(path)

        if not filename:
            modelName = self.modelName
            filename = f"{self.modelName}.pkl"

        else:
            modelName = filename
            filename = f"{filename}.pkl"

        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

        # #Guardamos el modelo LDA en el caso de haber usado LDA
        # if self.PRE_PROCES_PARAMS["lda"] == True:
        #     with open(f"lda_for_{filename}", 'wb') as file:
        #         pickle.dump(self.lda, file)

        #Guardamos los parámetros usados para entrenar el SVM
        file = open(f"{modelName}_preproces.json", "w")
        json.dump(self.PRE_PROCES_PARAMS , file)
        file.close

        file = open(f"{modelName}_fft.json", "w")
        json.dump(self.FFT_PARAMS , file)
        file.close

        os.chdir(actualFolder)

def main():

    """Empecemos"""

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"recordedEEG\WM\ses1")

    frecStimulus = np.array([6, 7, 8, 9])

    trials = 15
    fm = 200.
    window = 5 #sec
    samplePoints = int(fm*window)
    channels = 4

    filesRun1 = ["S3_R1_S2_E6","S3-R1-S1-E7", "S3-R1-S1-E8","S3-R1-S1-E9"]
    run1 = fa.loadData(path = path, filenames = filesRun1)
    filesRun2 = ["S3_R2_S2_E6","S3-R2-S1-E7", "S3-R2-S1-E8","S3-R2-S1-E9"]
    run2 = fa.loadData(path = path, filenames = filesRun2)

    #Filtering de EEG
    PRE_PROCES_PARAMS = {
                    'fm': fm,
                    'lfrec': 5.,
                    'hfrec': 30.,
                    'order': 6,
                    'sampling_rate': fm,
                    'bandStop': 50.,
                    'window': window,
                    'shiftLen':window
                    }

    resolution = np.round(fm/samplePoints, 4)

    FFT_PARAMS = {
                    'resolution': resolution,#0.2930,
                    'start_frequency': 5.0,
                    'end_frequency': 30.0,
                    'sampling_rate': fm
                    }

    def joinData(allData, stimuli, channels, samples, trials):
        joinedData = np.zeros((stimuli, channels, samples, trials))
        for i, sujeto in enumerate(allData):
            joinedData[i] = allData[sujeto]["eeg"][0,:,:,:trials]

        return joinedData #la forma de joinedData es [estímulos, canales, muestras, trials]

    run1JoinedData = joinData(run1, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)
    run2JoinedData = joinData(run2, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)

    trainSet = np.concatenate((run1JoinedData[:,:,:,:12], run2JoinedData[:,:,:,:12]), axis = 3)
    trainSet = trainSet[:,:1,:,:] #nos quedamos con los primeros dos canales

    trainSet = np.mean(trainSet, axis = 1) #promedio sobre los canales. Forma datos ahora [clases, samples, trials]

    #Aplicamos StandarScaler
    # for clase in range(clases):
    #     trainSet[clase] = StandardScaler().fit_transform(trainSet[clase]).reshape(samples, trials)

    nclases = trainSet.shape[0]
    nsamples = trainSet.shape[1]
    ntrials = trainSet.shape[2]

    #Restamos la media de la señal
    mean = trainSet.mean(axis = 1)
    for clase in range(nclases):
            for trial in range(ntrials):
                trainSet[clase, :, trial] =   trainSet[clase, :, trial] - mean[clase, trial]

    #Creo objeto SVMTrainingModule
    svm = SVMTrainingModule(trainSet, PRE_PROCES_PARAMS, FFT_PARAMS, frecStimulus=frecStimulus,
    nchannels = 1, nsamples = nsamples, ntrials = ntrials, modelName = "test")
    
    seed = np.random.randint(500) #iniciamos una semilla

    #Creamos modelo SVM
    modelo = svm.createSVM(kernel = "rbf", gamma = 0.05, C = 24, probability = True, randomSeed = seed)

    anchoVentana = int(fm*5) #fm * segundos
    ventana = windows.hamming #Usamos ventana Hamming

    sampleFrec, signalPSD  = svm.featuresExtraction(ventana = ventana, anchoVentana = 5, bw = 2.0, order = 6, axis = 1)

    metricas = svm.trainAndValidateSVM(clases = np.arange(0,len(frecStimulus)), test_size = 0.2, randomSeed = seed)
    print(metricas)

    ######################################################################
    ################## Buscando el mejor clasificador ####################
    ######################################################################

    hiperParams = {"kernels": ["linear", "rbf"],
        "gammaValues": [1e-2, 1e-1, 1, 1e+1, 1e+2, "scale", "auto"],
        "CValues": [8e-1,9e-1, 1, 1e2, 1e3]
        }

    clasificadoresSVM = {"linear": list(),
                    "rbf": list()
        }

    rbfResults = np.zeros((len(hiperParams["gammaValues"]), len(hiperParams["CValues"])))
    linearResults = list()

    seed = np.random.randint(100)

    for i, kernel in enumerate(hiperParams["kernels"]):

        if kernel != "linear":
            for j, gamma in enumerate(hiperParams["gammaValues"]):

                for k, C in enumerate(hiperParams["CValues"]):
                    #Instanciamos el modelo para los hipermarametros

                    svm = SVMTrainingModule(trainSet, PRE_PROCES_PARAMS, FFT_PARAMS, frecStimulus=frecStimulus,
                            nchannels = 1, nsamples = nsamples, ntrials = ntrials, modelName = "testSVMv2")

                    modelo = svm.createSVM(kernel = kernel, gamma = gamma, C = C, probability = True, randomSeed = seed)

                    sampleFrec, signalPSD  = svm.featuresExtraction(ventana = ventana, anchoVentana = 5, bw = 2.0, order = 6, axis = 1)

                    metricas = svm.trainAndValidateSVM(clases = np.arange(0,len(frecStimulus)), test_size = 0.2, randomSeed = seed) #entrenamos el modelo y obtenemos las métricas
                    accu = metricas["modelo_testSVMv2"]["val"]["Acc"]
                    rbfResults[j,k] = accu

                    clasificadoresSVM[kernel].append((C, gamma, svm, accu))
        else:
            for k, C in enumerate(hiperParams["CValues"]):

                svm = SVMTrainingModule(trainSet, PRE_PROCES_PARAMS, FFT_PARAMS, frecStimulus=frecStimulus,
                        nchannels = 1, nsamples = nsamples, ntrials = ntrials, modelName = "testSVMv2")

                modelo = svm.createSVM(kernel = kernel, C = C, probability = True, randomSeed = seed)

                sampleFrec, signalPSD  = svm.featuresExtraction(ventana = ventana, anchoVentana = 5, bw = 2.0, order = 6, axis = 1)

                metricas = svm.trainAndValidateSVM(clases = np.arange(0,len(frecStimulus)), test_size = 0.2, randomSeed = seed)

                accu = metricas["modelo_testSVMv2"]["val"]["Acc"]

                linearResults.append(accu)
                #predecimos con los datos en Xoptim

                clasificadoresSVM[kernel].append((C, svm, accu))

    plt.figure(figsize=(15,10))
    plt.imshow(rbfResults)
    plt.xlabel("Valor de C")
    plt.xticks(np.arange(len(hiperParams["CValues"])), hiperParams["CValues"])
    plt.ylabel("Valor de Gamma")
    plt.yticks(np.arange(len(hiperParams["gammaValues"])), hiperParams["gammaValues"])
    plt.colorbar()

    for i in range(rbfResults.shape[0]):
        for j in range(rbfResults.shape[1]):
            plt.text(j, i, "{:.2f}".format(rbfResults[i, j]), va='center', ha='center')
    plt.show()

    plt.plot([str(C) for C in hiperParams["CValues"]], np.asarray(linearResults)*100)
    plt.title("Accuracy para predicciones usando kernel 'linear'")
    plt.xlabel("Valor de C")
    plt.ylabel("Accuracy (%)")
    plt.show()

    #Selecciono dos clasificadores SVM
    modeloSVM1 = clasificadoresSVM["linear"][3][1]
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"models")
    modeloSVM1.saveModel(path, filename = "SVM_test_linear")
    modeloSVM1.saveTrainingSignalPSD(signalPSD.mean(axis = 2), path = path, filename = "SVM_test_linear")
    os.chdir(actualFolder)

    gamma = "scale"
    C = 100

    for values in clasificadoresSVM["rbf"]:
        if values[1] == gamma and values[0] == C:
            modeloSVM2 = values[2] #modelo 2 es un SVM con kernel = rbf

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder, "models")
    modeloSVM2.saveModel(path, filename = "SVM_test_rbf")
    modeloSVM2.saveTrainingSignalPSD(signalPSD.mean(axis = 2), path = path, filename = "SVM_test_rbf")
    os.chdir(actualFolder)

if __name__ == "__main__":
    main()
