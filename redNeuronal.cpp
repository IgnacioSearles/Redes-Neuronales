#include "redNeuronal.h"

#include <iostream>
#include <random>
#include <fstream>
#include <vector>

redNeuronal::redNeuronal(std::vector<int> mForma, funcionesDeActivacion mActivacion, funcionesDeError mError, funcionesDeNormalizacionDeVariables mNormalizacion, float mLambda)
{
    forma = mForma;
    funcionDeActivacion = mActivacion;
    funcionDeError = mError;
    funcionDeNormalizacion = mNormalizacion;
    lambda = mLambda;

    initWeights();
    initBias();
    initNodos();

    cambiarOffsetYMultiplicador(0.0001f, 1.0f);

    randomizarRed();
}

void redNeuronal::initWeights()
{
    weights.resize(forma.size() - 1);
    deltaWeights.resize(forma.size() - 1);

    for (int capa = 0; capa < forma.size() - 1; capa++)
    {
        weights.at(capa).resize(forma.at(capa + 1));
        deltaWeights.at(capa).resize(forma.at(capa + 1));

        for (int nodo = 0; nodo < forma.at(capa + 1); nodo++)
        {
            weights.at(capa).at(nodo).resize(forma.at(capa));
            deltaWeights.at(capa).at(nodo).resize(forma.at(capa));
        }
    }
}

void redNeuronal::initBias()
{
    biases.resize(forma.size() - 1);
    deltaBiases.resize(forma.size() - 1);

    for (int capa = 0; capa < forma.size() - 1; capa++)
    {
        biases.at(capa).resize(forma.at(capa + 1));
        deltaBiases.at(capa).resize(forma.at(capa + 1));
    }
}

void redNeuronal::initNodos()
{
    nodos.resize(forma.size());

    for (int capa = 0; capa < forma.size(); capa++)
    {
        nodos.at(capa).resize(forma.at(capa));
    }
}

void redNeuronal::randomizarRed()
{
    std::random_device generadorDeSemilla;
    std::mt19937 generador(generadorDeSemilla());
    std::uniform_real_distribution<float> distribucion(-1, 1);

    for (int capa = 0; capa < weights.size(); capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < weights.at(capa).size(); nodoDeSalida++)
        {
            for (int nodoDeEntrada = 0; nodoDeEntrada < weights.at(capa).at(nodoDeSalida).size(); nodoDeEntrada++)
            {
                weights.at(capa).at(nodoDeSalida).at(nodoDeEntrada) = distribucion(generador);
            }
        }
    }

    for (int capa = 0; capa < biases.size(); capa++)
    {
        for (int nodo = 0; nodo < biases.at(capa).size(); nodo++)
        {
            biases.at(capa).at(nodo) = distribucion(generador);
        }
    }
}

void redNeuronal::cambiarOffsetYMultiplicador(float offset, float multiplicador)
{
    OFFSET = offset;
    multiplicadorDeEntrenamiento = multiplicador;
}

std::vector<float> redNeuronal::predecir(const std::vector<float> &entradas)
{
    nodos.at(0) = entradas;

    propagar();

    return nodos.at(nodos.size() - 1);
}

void redNeuronal::propagar()
{
    for (int capa = 1; capa < nodos.size(); capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < nodos.at(capa).size(); nodoDeSalida++)
        {
            nodos.at(capa).at(nodoDeSalida) = 0;

            for (int nodoDeEntrada = 0; nodoDeEntrada < nodos.at(capa - 1).size(); nodoDeEntrada++)
            {
                nodos.at(capa).at(nodoDeSalida) += nodos.at(capa - 1).at(nodoDeEntrada) * weights.at(capa - 1).at(nodoDeSalida).at(nodoDeEntrada);
            }

            nodos.at(capa).at(nodoDeSalida) += biases.at(capa - 1).at(nodoDeSalida);
            nodos.at(capa).at(nodoDeSalida) = aplicarFuncionDeActivacion(nodos.at(capa).at(nodoDeSalida));
        }
    }
}

void redNeuronal::propagarDesdeVariable(int capaNodoVariable, int nodoDeSalidaVariable) {
    nodos.at(capaNodoVariable).at(nodoDeSalidaVariable) = 0;

    for (int nodoDeEntrada = 0; nodoDeEntrada < nodos.at(capaNodoVariable - 1).size(); nodoDeEntrada++)
    {
       nodos.at(capaNodoVariable).at(nodoDeSalidaVariable) += nodos.at(capaNodoVariable - 1).at(nodoDeEntrada) * weights.at(capaNodoVariable - 1).at(nodoDeSalidaVariable).at(nodoDeEntrada); 
    }

    nodos.at(capaNodoVariable).at(nodoDeSalidaVariable) += biases.at(capaNodoVariable - 1).at(nodoDeSalidaVariable);
    nodos.at(capaNodoVariable).at(nodoDeSalidaVariable) = aplicarFuncionDeActivacion(nodos.at(capaNodoVariable).at(nodoDeSalidaVariable));

    for (int capa = capaNodoVariable + 1; capa < nodos.size(); capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < nodos.at(capa).size(); nodoDeSalida++)
        {
            nodos.at(capa).at(nodoDeSalida) = 0;

            for (int nodoDeEntrada = 0; nodoDeEntrada < nodos.at(capa - 1).size(); nodoDeEntrada++)
            {
                nodos.at(capa).at(nodoDeSalida) += nodos.at(capa - 1).at(nodoDeEntrada) * weights.at(capa - 1).at(nodoDeSalida).at(nodoDeEntrada);
            }

            nodos.at(capa).at(nodoDeSalida) += biases.at(capa - 1).at(nodoDeSalida);
            nodos.at(capa).at(nodoDeSalida) = aplicarFuncionDeActivacion(nodos.at(capa).at(nodoDeSalida));
        }
    }
}

float redNeuronal::aplicarFuncionDeActivacion(const float &entrada)
{
    switch (funcionDeActivacion)
    {
    case sigmoid:
        return 1.0f / (1.0f + exp(-entrada));
    }
    return -1.0f;
}

void redNeuronal::entrenar(const std::vector<std::vector<float>> &ejemplos, const std::vector<std::vector<float>> &objetivos)
{
    for (int ejemplo = 0; ejemplo < ejemplos.size(); ejemplo++)
    {
        nodos.at(0) = ejemplos.at(ejemplo);
        objetivosActuales = objetivos.at(ejemplo);

        adjustarVariablesSegunEjemplo();
        std::cout << "\r" << ejemplo + 1 << "/" << ejemplos.size();
    }
}

void redNeuronal::mezclarEjemplos(std::vector<std::vector<float>> &ejemplos, std::vector<std::vector<float>> &objetivos)
{
    std::random_device generadorDeSemilla;
    std::mt19937 generador(generadorDeSemilla());
    std::uniform_int_distribution<int> distribucion(0, ejemplos.size() - 1);

    for (int ejemplo = 0; ejemplo < ejemplos.size(); ejemplo++)
    {
        int nuevaPosicion = distribucion(generador);

        std::swap(ejemplos.at(ejemplo), ejemplos.at(nuevaPosicion));
        std::swap(objetivos.at(ejemplo), objetivos.at(nuevaPosicion));
    }
}

void redNeuronal::adjustarVariablesSegunEjemplo()
{
    propagar();
    errorPrediccionInicial = calcularError();

    for (int capa = 0; capa < weights.size(); capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < weights.at(capa).size(); nodoDeSalida++)
        {
            for (int nodoDeEntrada = 0; nodoDeEntrada < weights.at(capa).at(nodoDeSalida).size(); nodoDeEntrada++)
            {
                calcularGradienteSegunError(weights.at(capa).at(nodoDeSalida).at(nodoDeEntrada), deltaWeights.at(capa).at(nodoDeSalida).at(nodoDeEntrada), capa + 1, nodoDeSalida);
            }
        }
    }

    for (int capa = 0; capa < weights.size(); capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < weights.at(capa).size(); nodoDeSalida++)
        {
            for (int nodoDeEntrada = 0; nodoDeEntrada < weights.at(capa).at(nodoDeSalida).size(); nodoDeEntrada++)
            {
                weights.at(capa).at(nodoDeSalida).at(nodoDeEntrada) += deltaWeights.at(capa).at(nodoDeSalida).at(nodoDeEntrada);
            }
        }
    }

    propagar();
    errorPrediccionInicial = calcularError();

    for (int capa = 0; capa < biases.size(); capa++)
    {
        for (int nodo = 0; nodo < biases.at(capa).size(); nodo++)
        {
            calcularGradienteSegunError(biases.at(capa).at(nodo), deltaBiases.at(capa).at(nodo), capa + 1, nodo);
        }
    }

    for (int capa = 0; capa < biases.size(); capa++)
    {
        for (int nodo = 0; nodo < biases.at(capa).size(); nodo++)
        {
            biases.at(capa).at(nodo) += deltaBiases.at(capa).at(nodo);
        }
    }
}

void redNeuronal::calcularGradienteSegunError(float &variable, float &deltaVariable, int capaVariable, int nodoDeSalidaVariable)
{
    variable += OFFSET;
    propagarDesdeVariable(capaVariable, nodoDeSalidaVariable);
    float errorPrediccionActual = calcularError();
    variable -= OFFSET;

    deltaVariable = ((errorPrediccionActual - errorPrediccionInicial) / OFFSET) * -multiplicadorDeEntrenamiento;
}

float redNeuronal::calcularError()
{
    float error = 0;
    for (int nodo = 0; nodo < nodos.at(nodos.size() - 1).size(); nodo++)
    {
        error += aplicarFuncionDeError(objetivosActuales.at(nodo), nodos.at(nodos.size() - 1).at(nodo));
    }

    error = aplicarFuncionDePromedioError(error);
    error += aplicarFuncionDeNormalizacionDeVariables();
    return error;
}

float redNeuronal::aplicarFuncionDeError(const float &actual, const float &prediccion)
{
    switch (funcionDeError)
    {
    case meanSquared:
        return (actual - prediccion) * (actual - prediccion);
    case crossEntropy:
        return actual * log(prediccion) + (1 - actual) * log(1 - prediccion);
    }
    return -1.0f;
}

float redNeuronal::aplicarFuncionDePromedioError(const float &error)
{
    switch (funcionDeError)
    {
    case meanSquared:
        return error * (1.0f / (2.0f * nodos.at(nodos.size() - 1).size()));
    case crossEntropy:
        return error * -(1.0f / nodos.at(nodos.size() - 1).size());
    }
    return -1.0f;
}

float redNeuronal::aplicarFuncionDeNormalizacionDeVariables()
{
    switch (funcionDeNormalizacion)
    {
    case ninguna:
        return 0;
    case weightNormalizacion:
        float sumaWeights = 0;
        int numWeights = 0;
        for (int capa = 0; capa < weights.size(); capa++)
        {
            for (int nodoDeSalida = 0; nodoDeSalida < weights.at(capa).size(); nodoDeSalida++)
            {
                for (int nodoDeEntrada = 0; nodoDeEntrada < weights.at(capa).at(nodoDeSalida).size(); nodoDeEntrada++)
                {
                    sumaWeights += weights.at(capa).at(nodoDeSalida).at(nodoDeEntrada) * weights.at(capa).at(nodoDeSalida).at(nodoDeEntrada);
                    numWeights++;
                }
            }
        }
        return (lambda / (2 * numWeights)) * sumaWeights;
    }
}

void redNeuronal::guardarRedEnArchivo(const char *archivo)
{
    std::ofstream archivoRedNeuronal;
    archivoRedNeuronal.open(archivo, std::ofstream::binary);

    for (int capa = 0; capa < weights.size(); capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < weights.at(capa).size(); nodoDeSalida++)
        {
            for (int nodoDeEntrada = 0; nodoDeEntrada < weights.at(capa).at(nodoDeSalida).size(); nodoDeEntrada++)
            {
                archivoRedNeuronal.write((char *)&weights.at(capa).at(nodoDeSalida).at(nodoDeEntrada), sizeof(float));
            }
        }
    }

    for (int capa = 0; capa < biases.size(); capa++)
    {
        for (int nodo = 0; nodo < biases.at(capa).size(); nodo++)
        {
            archivoRedNeuronal.write((char *)&biases.at(capa).at(nodo), sizeof(float));
        }
    }

    archivoRedNeuronal.close();
}

void redNeuronal::abrirRedDeArchivo(const char *archivo)
{
    std::ifstream archivoRedNeuronal;
    archivoRedNeuronal.open(archivo, std::ifstream::binary);

    for (int capa = 0; capa < weights.size(); capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < weights.at(capa).size(); nodoDeSalida++)
        {
            for (int nodoDeEntrada = 0; nodoDeEntrada < weights.at(capa).at(nodoDeSalida).size(); nodoDeEntrada++)
            {
                archivoRedNeuronal.read((char *)&weights.at(capa).at(nodoDeSalida).at(nodoDeEntrada), sizeof(float));
            }
        }
    }

    for (int capa = 0; capa < biases.size(); capa++)
    {
        for (int nodo = 0; nodo < biases.at(capa).size(); nodo++)
        {
            archivoRedNeuronal.read((char *)&biases.at(capa).at(nodo), sizeof(float));
        }
    }

    archivoRedNeuronal.close();
}

void redNeuronal::imprimirInfoPorConsola()
{
    std::cout << "-- Red Neuronal --" << std::endl
              << std::endl;

    std::cout << "Forma red: " << std::endl;
    for (int capa = 0; capa < forma.size(); capa++)
        std::cout << "- " << forma.at(capa) << std::endl;
    std::cout << std::endl;

    std::cout << "Weights red: " << std::endl;
    for (int capa = 0; capa < weights.size(); capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < weights.at(capa).size(); nodoDeSalida++)
        {
            for (int nodoDeEntrada = 0; nodoDeEntrada < weights.at(capa).at(nodoDeSalida).size(); nodoDeEntrada++)
            {
                std::cout << std::fixed << "Weight[" << capa + 1 << ", " << nodoDeSalida << ", " << nodoDeEntrada << "]: " << weights.at(capa).at(nodoDeSalida).at(nodoDeEntrada) << std::endl;
            }
        }
    }

    std::cout << "Biases red: " << std::endl;
    for (int capa = 0; capa < biases.size(); capa++)
    {
        for (int nodo = 0; nodo < biases.at(capa).size(); nodo++)
        {
            std::cout << "Bias[" << capa + 1 << ", " << nodo << "]: " << biases.at(capa).at(nodo) << std::endl;
        }
    }
}