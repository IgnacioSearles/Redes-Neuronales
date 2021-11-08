#include "../redNeuronal.h"
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Por favor pasar nombre de archivo y accion (entrenar o abrir) como argumento." << std::endl;
        return 0;
    }

    std::string nombreArchivo = argv[1];
    std::string accion = argv[2];

    std::vector<int> forma = {2, 5, 1};
    redNeuronal RN(forma, sigmoid, crossEntropy);

    if (accion == "entrenar")
    {
        RN.imprimirInfoPorConsola();

        std::cout << "\n\nEntrenando red....\n\n";

        RN.cambiarOffsetYMultiplicador(0.0001f, 0.1f);

        std::vector<std::vector<float>> ejemplos_xor = {{1, 0}, {0, 1}, {0, 0}, {1, 1}};
        std::vector<std::vector<float>> objetivos_xor = {{1}, {1}, {0}, {0}};

        for (int epoch = 0; epoch < 500; epoch++)
        {
            RN.mezclarEjemplos(ejemplos_xor, objetivos_xor);
            std::cout << "Epoch " << epoch << "/500: " << std::endl;

            RN.entrenar(ejemplos_xor, objetivos_xor);
            std::cout << std::endl;
        }

        RN.guardarRedEnArchivo(nombreArchivo.c_str());

        std::cout << "\nRed guardada en " << nombreArchivo << std::endl << std::endl;
    }
    else {
        RN.imprimirInfoPorConsola();

        std::cout << "\n\nAbriendo red de " << nombreArchivo << std::endl << std::endl;

        RN.abrirRedDeArchivo(nombreArchivo.c_str());
    }

    std::cout << "Probando casos: \n\n";
    std::vector<std::vector<float>> test = {{1, 0}, {1, 1}, {0, 0}, {0, 1}};

    for (int ejem = 0; ejem < 4; ejem++)
    {
        std::vector<float> prediccion = RN.predecir(test.at(ejem));
        std::cout << "Prediccion (" << test.at(ejem).at(0) << ", " << test.at(ejem).at(1) << "): " << prediccion.at(0) << std::endl;
    }

    return 0;
}