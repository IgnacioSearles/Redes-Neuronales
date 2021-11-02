#include "../redNeuronal.h"

#include <iostream>
#include <fstream>

#define OFFSET_ARCHIVO_IMG 16
#define OFFSET_ARCHIVO_OBJETIVOS 8

float *cargarInfoMNIST(const int offset, const char *nombreDeArchivo)
{
    std::ifstream archivoDeEntrada;
    archivoDeEntrada.open(nombreDeArchivo);

    archivoDeEntrada.seekg(0, archivoDeEntrada.end);
    int largoArchivo = archivoDeEntrada.tellg();
    archivoDeEntrada.seekg(0, archivoDeEntrada.beg);

    unsigned char *bufferEntrada = (unsigned char *)malloc(largoArchivo - offset);

    archivoDeEntrada.seekg(offset);
    archivoDeEntrada.read((char *)bufferEntrada, largoArchivo - offset);
    archivoDeEntrada.close();

    float *infoMNIST = (float *)malloc((largoArchivo - offset) * sizeof(float));
    for (int val = 0; val < largoArchivo - offset; val++) {
        infoMNIST[val] = bufferEntrada[val];
    }

    free(bufferEntrada);
    return infoMNIST;
}

void objetivosAOneShot(float* objetivos, int size) {
    float* objetivosOneShot;
    objetivosOneShot = (float*)malloc(size * 10 * sizeof(float));
    for (int objetivo = 0; objetivo < size; objetivo++) {
        for (int nodo = 0; nodo < 10; nodo++) {
            if (objetivos[objetivo] == nodo) objetivosOneShot[objetivo * 10 + nodo] = 1;
            else objetivosOneShot[objetivo * 10 + nodo] = 0;
        }
    }
    free(objetivos);
    objetivos = objetivosOneShot;
}

int obtenerIndiceDelMaximo(float* buffer) {
    float max = 0;
    int indiceMax;
    for (int i = 0; i < 10; i++) {
        if (buffer[i] > max) {
            max = buffer[i];
            indiceMax = i;
        }
    }
    return indiceMax;
}

float calcularPrecision(redNeuronal &RN, float* ejemplos, float* objetivos, int numEjemplos) {
    int correctos = 0;
    for (int ejemplo = 0; ejemplo < numEjemplos; ejemplo++) {
        float *ejemploBuffer = (float*) malloc(784 * sizeof(float));
        memcpy(ejemploBuffer, ejemplos + ejemplo * 784, 784 * sizeof(float));

        float* prediccion = RN.predecir(ejemploBuffer);
        int indicePrediccion = obtenerIndiceDelMaximo(prediccion);
        if (objetivos[ejemplo * 10 + indicePrediccion] == 1) correctos += 1;

        free(ejemploBuffer);
        free(prediccion);
    }
    return ((float)correctos / numEjemplos) * 100.0f;
}

int main()
{
    int forma[] = {784, 8, 10};
    redNeuronal RN(forma, sizeof(forma) / sizeof(int), sigmoid, crossEntropy);
    RN.cambiarOffsetYMultiplicador(0.0001f, 0.1f);

    imagenesTrain = cargarInfoMNIST(OFFSET_ARCHIVO_IMG, "mnistTrain60KImgs.bytes");
    float *objetivosTrain = cargarInfoMNIST(OFFSET_ARCHIVO_OBJETIVOS, "mnistTrain60KLabels.bytes");
    objetivosAOneShot(objetivosTrain, 60000);

    float *imagenesTest = cargarInfoMNIST(OFFSET_ARCHIVO_IMG, "mnistTest10KImgs.bytes");
    float *objetivosTest = cargarInfoMNIST(OFFSET_ARCHIVO_OBJETIVOS, "mnistTest10KLabels.bytes");
    objetivosAOneShot(objetivosTest, 10000);

    for (int epoch = 0; epoch < 10; epoch++) {
        std::cout << "Epoch " << epoch + 1 << "/10: " << std::endl;
        RN.entrenar(imagenesTest, objetivosTest, 10000);
        std::cout << std::endl;
    }
    std::cout << "\nPrecision: " << calcularPrecision(RN, imagenesTest, objetivosTest, 10000) << "%" << std::endl << std::endl;

    RN.guardarRedEnArchivo("mnist.red");

    free(imagenesTrain);
    free(objetivosTrain);
    free(imagenesTest);
    free(objetivosTest);

    return 0;
}