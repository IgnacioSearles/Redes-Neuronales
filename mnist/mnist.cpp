#include "../redNeuronal.h"

#include <iostream>
#include <fstream>

#define OFFSET_ARCHIVO_IMG 16
#define OFFSET_ARCHIVO_OBJETIVOS 8

#define SIZE_ARCHIVO_IMG 784
#define SIZE_ARCHIVO_OBJETIVOS 1

float normalizarValor(unsigned char valor, float numNormalizador) {
    return (float)valor / numNormalizador;
}

std::vector<std::vector<float>> cargarInfoMNIST(const int sizeDatos, const int offset, float numNormalizador, const char *nombreDeArchivo)
{
    std::ifstream archivoDeEntrada;
    archivoDeEntrada.open(nombreDeArchivo, std::ifstream::binary);

    archivoDeEntrada.seekg(0, archivoDeEntrada.end);
    int largoArchivo = archivoDeEntrada.tellg();
    archivoDeEntrada.seekg(0, archivoDeEntrada.beg);

    if (offset > 0) archivoDeEntrada.seekg(offset);

    std::vector<std::vector<float>> salida;
    salida.resize((largoArchivo - offset) / sizeDatos);
    for (int ejemplo = 0; ejemplo < (largoArchivo - offset) / sizeDatos; ejemplo++) {
        for (int data = 0; data < sizeDatos; data++){
            unsigned char valor;
            archivoDeEntrada.read((char*)&valor, sizeof(unsigned char));
            salida.at(ejemplo).push_back(normalizarValor(valor, numNormalizador));
        }
    }

    archivoDeEntrada.close();

    return salida;
}

std::vector<float> objetivoAOneShot(const std::vector<float>& objetivo) {
    std::vector<float> salida;
    for (int nodo = 0; nodo < 10; nodo++) {
        if (objetivo.at(0) == nodo) salida.push_back(1);
        else salida.push_back(0);
    }
    return salida;
}

void objetivosAOneShot(std::vector<std::vector<float>>& objetivos) {
    for (int objetivo = 0; objetivo < objetivos.size(); objetivo++) {
        objetivos.at(objetivo) = objetivoAOneShot(objetivos.at(objetivo));
    }
}

int obtenerIndiceDelMaximo(const std::vector<float>& buffer) {
    float max = 0;
    int indiceMax;
    for (int i = 0; i < 10; i++) {
        if (buffer.at(i) > max) {
            max = buffer.at(i);
            indiceMax = i;
        }
    }
    return indiceMax;
}

float calcularPrecision(redNeuronal &RN, const std::vector<std::vector<float>>& ejemplos, const std::vector<std::vector<float>>& objetivos) {
    int correctos = 0;
    for (int ejemplo = 0; ejemplo < ejemplos.size(); ejemplo++) {
        std::vector<float> prediccion = RN.predecir(ejemplos.at(ejemplo));
        int indicePrediccion = obtenerIndiceDelMaximo(prediccion);
        if (objetivos.at(ejemplo).at(indicePrediccion) == 1) correctos += 1;
    }
    return ((float)correctos / ejemplos.size()) * 100.0f;
}

int main()
{
    std::vector<int> forma = {784, 16, 10};
    redNeuronal RN(forma, sigmoid, crossEntropy, weightNormalizacion, 0.8f);
    RN.cambiarOffsetYMultiplicador(0.0001f, 0.01f);
    RN.abrirRedDeArchivo("mnist.red");

    std::vector<std::vector<float>> imagenesDePrueba = cargarInfoMNIST(SIZE_ARCHIVO_IMG, OFFSET_ARCHIVO_IMG, 255.0f, "mnistTrain60KImgs.bytes");
    std::vector<std::vector<float>> objetivosDePrueba  = cargarInfoMNIST(SIZE_ARCHIVO_OBJETIVOS, OFFSET_ARCHIVO_OBJETIVOS, 1.0f, "mnistTrain60KLabels.bytes");
    objetivosAOneShot(objetivosDePrueba);

    std::vector<std::vector<float>> imagenesDeEntrenamiento = cargarInfoMNIST(SIZE_ARCHIVO_IMG, OFFSET_ARCHIVO_IMG, 255.0f, "mnistTest10KImgs.bytes");
    std::vector<std::vector<float>> objetivosDeEntrenamiento  = cargarInfoMNIST(SIZE_ARCHIVO_OBJETIVOS, OFFSET_ARCHIVO_OBJETIVOS, 1.0f, "mnistTest10KLabels.bytes");
    objetivosAOneShot(objetivosDeEntrenamiento);

    std::cout << "\nPrecision Inicial: " << calcularPrecision(RN, imagenesDeEntrenamiento, objetivosDeEntrenamiento) << "%" << std::endl << std::endl;
    for (int epoch = 0; epoch < 5; epoch++) {
        RN.mezclarEjemplos(imagenesDeEntrenamiento, objetivosDeEntrenamiento);
        std::cout << "Epoch " << epoch + 1 << "/5: " << std::endl;
        RN.entrenar(imagenesDeEntrenamiento, objetivosDeEntrenamiento);
        std::cout << std::endl;
        std::cout << "\nPrecision: " << calcularPrecision(RN, imagenesDePrueba, objetivosDePrueba) << "%" << std::endl << std::endl;
        RN.guardarRedEnArchivo("mnist.red");
    }

    return 0;
}