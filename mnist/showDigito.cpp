#include <iostream>
#include <fstream>

#include <vector>

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

int main()
{
    std::vector<std::vector<float>> imagenesDeEntrenamiento = cargarInfoMNIST(SIZE_ARCHIVO_IMG, OFFSET_ARCHIVO_IMG, 255.0f, "mnistTest10KImgs.bytes");
    std::vector<std::vector<float>> objetivosDeEntrenamiento  = cargarInfoMNIST(SIZE_ARCHIVO_OBJETIVOS, OFFSET_ARCHIVO_OBJETIVOS, 1.0f, "mnistTest10KLabels.bytes");
    objetivosAOneShot(objetivosDeEntrenamiento);

    for (int digito = 0; digito < 5; digito++) {
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                std::cout << (imagenesDeEntrenamiento.at(digito).at(y * 28 + x) > 0) ? '#' : ' ';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        for (int i = 0; i < 10; i++) {
            std::cout << objetivosDeEntrenamiento.at(digito).at(i) << std::endl;
        }
    }

    return 0;
}