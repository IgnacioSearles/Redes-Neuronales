#pragma once

#include <vector>

enum funcionesDeActivacion {sigmoid};
enum funcionesDeError {meanSquared, crossEntropy};
enum funcionesDeNormalizacionDeVariables {ninguna, weightNormalizacion};

class redNeuronal
{
public:
    redNeuronal(std::vector<int> mForma, funcionesDeActivacion mActivacion, funcionesDeError mError, funcionesDeNormalizacionDeVariables mNormalizacion = ninguna, float mLambda = 0.1f);

    void imprimirInfoPorConsola();
    void cambiarOffsetYMultiplicador(float offset, float multiplicador);

    std::vector<float> predecir(const std::vector<float> &entradas);

    void entrenar(const std::vector<std::vector<float>>& ejemplos, const std::vector<std::vector<float>>& objetivos);
    void mezclarEjemplos(std::vector<std::vector<float>>& ejemplos, std::vector<std::vector<float>>& objetivos);

    void guardarRedEnArchivo(const char* archivo);
    void abrirRedDeArchivo(const char* archivo);
private:
    void initWeights();
    void initBias();
    void initNodos();

    void randomizarRed();

    void propagar();
    void propagarDesdeVariable(int capaVariable, int nodoDeSalidaVariable);
    float aplicarFuncionDeActivacion(const float& entrada);

    float calcularError();
    float aplicarFuncionDeError(const float& actual, const float& prediccion);
    float aplicarFuncionDePromedioError(const float& error);
    float aplicarFuncionDeNormalizacionDeVariables();

    void adjustarVariablesSegunEjemplo();
    void calcularGradienteSegunError(float &variable, float &deltaVariable, int capaVariable, int nodoDeSalidaVariable);

    float OFFSET, multiplicadorDeEntrenamiento;

    std::vector<int> forma;

    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> biases;
    std::vector<std::vector<float>> nodos;

    std::vector<float> objetivosActuales;
    std::vector<std::vector<std::vector<float>>> deltaWeights;
    std::vector<std::vector<float>> deltaBiases;

    float errorPrediccionInicial;

    float lambda;

    funcionesDeActivacion funcionDeActivacion;
    funcionesDeNormalizacionDeVariables funcionDeNormalizacion;
    funcionesDeError funcionDeError;
};