#include "redNeuronalGPU.h"

#include <iostream>
#include <random>
#include <fstream>

__global__ void realizarCalculosWeightsEnGPU(float* weights, float* nodos, int capa, int numWeights, int sizeAtCapa, int* prevNodosAtCapa, int* prevWeightsAtCapa) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    int paso = blockDim.x * gridDim.x;

    for (int weight = indice; weight < numWeights; weight += paso)
    {
        int nodoDeSalida = weight % sizeAtCapa;
        int nodoDeEntrada = weight / sizeAtCapa;
        float temp =  nodos[prevNodosAtCapa[capa - 1] + nodoDeEntrada] * weights[prevWeightsAtCapa[capa] + nodoDeEntrada * sizeAtCapa + nodoDeSalida];
        atomicAdd(&nodos[prevNodosAtCapa[capa] + nodoDeSalida], temp);                                                
    }
}

redNeuronal::redNeuronal(int mForma[], int mNumCapas, funcionesDeActivacion mActivacion, funcionesDeError mError)
{
    forma = mForma;
    numCapas = mNumCapas;
    funcionDeActivacion = mActivacion;
    funcionDeError = mError;

    initWeights();
    initBias();
    initNodos();

    objetivosActuales = (float*)malloc(forma[numCapas - 1] * sizeof(float));

    cambiarOffsetYMultiplicador(0.0001f, 1.0f);

    randomizarRed();
}

void redNeuronal::initWeights()
{
    numWeights = 0;

    for (int capa = 1; capa < numCapas; capa++)
    {
        numWeights += forma[capa - 1] * forma[capa];
    }
    cudaMallocManaged(&weights, numWeights * sizeof(float));
    deltaWeights = (float *)malloc(numWeights * sizeof(float));

    cudaMallocManaged(&prevWeightsAtCapa, numCapas * sizeof(int));
    prevWeightsAtCapa[0] = 0;
    prevWeightsAtCapa[1] = 0;
    for (int capa = 2; capa < numCapas; capa++)
    {
        prevWeightsAtCapa[capa] = forma[capa - 2] * forma[capa - 1] + prevWeightsAtCapa[capa - 1];
    }
}

void redNeuronal::initBias()
{
    numBias = 0;

    for (int capa = 1; capa < numCapas; capa++)
    {
        numBias += forma[capa];
    }
    biases = (float *)malloc(numBias * sizeof(float));
    deltaBiases = (float *)malloc(numBias * sizeof(float));

    prevBiasAtCapa = (int *)malloc(numCapas * sizeof(int));
    prevBiasAtCapa[0] = 0;
    prevBiasAtCapa[1] = 0;
    for (int capa = 2; capa < numCapas; capa++)
    {
        prevBiasAtCapa[capa] = forma[capa - 1] + prevBiasAtCapa[capa - 1];
    }
}

void redNeuronal::initNodos()
{
    numNodos = 0;

    for (int capa = 0; capa < numCapas; capa++)
    {
        numNodos += forma[capa];
    }
    cudaMallocManaged(&nodos, numNodos * sizeof(float));

    cudaMallocManaged(&prevNodosAtCapa, numCapas * sizeof(int));
    prevNodosAtCapa[0] = 0;
    for (int capa = 1; capa < numCapas; capa++)
    {
        prevNodosAtCapa[capa] = forma[capa - 1] + prevNodosAtCapa[capa - 1];
    }
}

void redNeuronal::randomizarRed()
{
    std::random_device generadorDeSemilla;
    std::mt19937 generador(generadorDeSemilla());
    std::uniform_real_distribution<float> distribucion(-1, 1);

    for (int weight = 0; weight < numWeights; weight++)
    {
        weights[weight] = distribucion(generador);
    }

    for (int bias = 0; bias < numBias; bias++)
    {
        biases[bias] = distribucion(generador);
    }
}

void redNeuronal::imprimirInfoPorConsola()
{
    std::cout << "Forma red: " << std::endl;
    for (int capa = 0; capa < numCapas; capa++)
        std::cout << "- " << forma[capa] << std::endl;
    std::cout << std::endl;

    std::cout << "Weights red: " << std::endl;
    for (int capa = 1; capa < numCapas; capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < forma[capa]; nodoDeSalida++)
        {
            for (int nodoDeEntrada = 0; nodoDeEntrada < forma[capa - 1]; nodoDeEntrada++)
            {
                std::cout << std::fixed << "Weight[" << capa << ", " << nodoDeSalida << ", " << nodoDeEntrada << "]: " << weights[prevWeightsAtCapa[capa] + nodoDeEntrada * forma[capa] + nodoDeSalida] << std::endl;
            }
        }
    }

    std::cout << "Biases red: " << std::endl;
    for (int capa = 1; capa < numCapas; capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < forma[capa]; nodoDeSalida++)
        {
            std::cout << "Bias[" << capa << ", " << nodoDeSalida << "]: " << biases[prevBiasAtCapa[capa] + nodoDeSalida] << std::endl;
        }
    }
}

void redNeuronal::cambiarOffsetYMultiplicador(float offset, float multiplicador) {
    OFFSET = offset;
    multiplicadorDeEntrenamiento = multiplicador;
}

float *redNeuronal::predecir(float *entradas)
{
    memcpy(nodos, entradas, forma[0] * sizeof(float));

    propagarEnGPU();

    float *salida = new float[forma[numCapas - 1]];
    memcpy(salida, nodos + prevNodosAtCapa[numCapas - 1], forma[numCapas - 1] * sizeof(float));
    return salida;
}

void redNeuronal::propagarEnGPU()
{
    for (int capa = 1; capa < numCapas; capa++)
    {
        for (int nodoDeSalida = 0; nodoDeSalida < forma[capa]; nodoDeSalida++)
        {
            nodos[prevNodosAtCapa[capa] + nodoDeSalida] = 0;
        }

        int bloqueSize = 256;
        int numBloques = (forma[capa] * forma[capa - 1] + bloqueSize - 1) / bloqueSize;

        realizarCalculosWeightsEnGPU<<<numBloques, bloqueSize>>>(weights, nodos, capa, forma[capa] * forma[capa - 1], forma[capa], prevNodosAtCapa, prevWeightsAtCapa);
        cudaDeviceSynchronize();

        for (int nodoDeSalida = 0; nodoDeSalida < forma[capa]; nodoDeSalida++)
        {
            nodos[prevNodosAtCapa[capa] + nodoDeSalida] += biases[prevBiasAtCapa[capa] + nodoDeSalida];
            nodos[prevNodosAtCapa[capa] + nodoDeSalida] = aplicarFuncionDeActivacion(nodos[prevNodosAtCapa[capa] + nodoDeSalida]);
        }
    }
}

float redNeuronal::aplicarFuncionDeActivacion(const float& entrada) {
    switch (funcionDeActivacion)
    {
    case sigmoid:
        return 1.0f / (1.0f + exp(-entrada));
    }
    return -1.0f;
}

void redNeuronal::entrenar(float *ejemplos, float* objetivos, int numEjemplos) {
    //mezclarEjemplos(ejemplos, objetivos, numEjemplos);

    for (int ejemplo = 0; ejemplo < numEjemplos; ejemplo++) {
        memcpy(nodos, ejemplos + ejemplo * forma[0], forma[0] * sizeof(float));
        memcpy(objetivosActuales, objetivos + ejemplo * forma[numCapas - 1], forma[numCapas - 1] * sizeof(float));
                
        adjustarVariablesSegunEjemplo();

        std::cout << "\r" << ejemplo << "/" << numEjemplos;
    }
}

void redNeuronal::mezclarEjemplos(float* ejemplos, float* objetivos, int numEjemplos) {
    std::random_device generadorDeSemilla;
    std::mt19937 generador(generadorDeSemilla());
    std::uniform_int_distribution<int> distribucion(0, numEjemplos - 1);

    for (int ejemplo = 0; ejemplo < numEjemplos; ejemplo++) {
        int nuevaPos = distribucion(generador);
        float* tempEntradas = (float*) malloc(forma[0] * sizeof(float));
        memcpy(tempEntradas, ejemplos + nuevaPos * forma[0], forma[0] * sizeof(float));
        memcpy(ejemplos + nuevaPos * forma[0], ejemplos + ejemplo * forma[0], forma[0] * sizeof(float));
        memcpy(ejemplos + ejemplo * forma[0], tempEntradas, forma[0] * sizeof(float));

        float* tempSalidas = (float*) malloc(forma[numCapas - 1] * sizeof(float));
        memcpy(tempSalidas, objetivos + nuevaPos * forma[numCapas - 1], forma[numCapas - 1] * sizeof(float));
        memcpy(objetivos + nuevaPos * forma[numCapas - 1], objetivos + ejemplo * forma[numCapas - 1], forma[numCapas - 1] * sizeof(float));
        memcpy(objetivos + ejemplo * forma[numCapas - 1], tempSalidas, forma[numCapas - 1] * sizeof(float));
    }
}

void redNeuronal::adjustarVariablesSegunEjemplo() {
    errorPrediccionInicial = calcularError();

    for (int weight = 0; weight < numWeights; weight++) {
        calcularGradienteSegunError(weights[weight], deltaWeights[weight]);
    }

    for (int weight = 0; weight < numWeights; weight++) {
        weights[weight] += deltaWeights[weight];
    }

    errorPrediccionInicial = calcularError();

    for (int bias = 0; bias < numBias; bias++) {
        calcularGradienteSegunError(biases[bias], deltaBiases[bias]);
    }

    for (int bias = 0; bias < numBias; bias++) {
        biases[bias] += deltaBiases[bias];
    }
}

void redNeuronal::calcularGradienteSegunError(float& variable, float& deltaVariable) {
    variable += OFFSET;
    float errorPrediccionActual = calcularError();
    variable -= OFFSET;

    deltaVariable = ((errorPrediccionActual - errorPrediccionInicial) / OFFSET) * -multiplicadorDeEntrenamiento;
}

float redNeuronal::calcularError() {
    propagarEnGPU();

    float error = 0;
    for (int nodo = 0; nodo < forma[numCapas - 1]; nodo++) {
        error +=  aplicarFuncionDeError(objetivosActuales[nodo], nodos[prevNodosAtCapa[numCapas - 1] + nodo]);
    }

    error = aplicarFuncionDePromedioError(error);
    return error;
}

float redNeuronal::aplicarFuncionDeError(const float& actual, const float& prediccion) {
    switch (funcionDeError)
    {
    case meanSquared:
        return (actual - prediccion) * (actual - prediccion);
    case crossEntropy:
        return actual * log(prediccion) + (1 - actual) * log(1 - prediccion);
    }
    return -1.0f;
}

float redNeuronal::aplicarFuncionDePromedioError(const float& error) {
    switch (funcionDeError)
    {
    case meanSquared:
        return error * (1.0f / (2.0f * forma[numCapas - 1]));
    case crossEntropy:
        return error * -(1.0f / forma[numCapas - 1]);
    }
    return -1.0f;
}

void redNeuronal::guardarRedEnArchivo(const char* archivo) {
    std::ofstream archivoRedNeuronal;
    archivoRedNeuronal.open(archivo);
    archivoRedNeuronal.write((char*)weights, numWeights * sizeof(float));
    archivoRedNeuronal.write((char*)biases, numBias * sizeof(float));
    archivoRedNeuronal.close();
}

void redNeuronal::abrirRedDeArchivo(const char* archivo) {
    std::ifstream archivoRedNeuronal;
    archivoRedNeuronal.open(archivo);
    archivoRedNeuronal.read((char*)weights, numWeights * sizeof(float));
    archivoRedNeuronal.read((char*)biases, numBias * sizeof(float));
    archivoRedNeuronal.close();
}

redNeuronal::~redNeuronal()
{
    cudaFree(prevWeightsAtCapa);
    free(prevBiasAtCapa);
    cudaFree(prevNodosAtCapa);

    cudaFree(weights);
    free(biases);
    cudaFree(nodos);

    free(deltaWeights);
    free(deltaBiases);
    free(objetivosActuales);
}