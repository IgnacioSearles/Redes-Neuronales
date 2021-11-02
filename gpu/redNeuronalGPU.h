#pragma once

enum funcionesDeActivacion {sigmoid};
enum funcionesDeError {meanSquared, crossEntropy};

class redNeuronal
{
public:
    redNeuronal(int mForma[], int mNumCapas, funcionesDeActivacion mActivacion, funcionesDeError mError);
    ~redNeuronal();

    void imprimirInfoPorConsola();
    void cambiarOffsetYMultiplicador(float offset, float multiplicador);

    float *predecir(float *entradas);

    void entrenar(float *ejemplos, float* objetivos, int numEjemplos);

    void guardarRedEnArchivo(const char* archivo);
    void abrirRedDeArchivo(const char* archivo);
private:
    void initWeights();
    void initBias();
    void initNodos();

    void randomizarRed();

    void propagarEnGPU();
    float aplicarFuncionDeActivacion(const float& entrada);

    void mezclarEjemplos(float *ejemplos, float* objetivos, int numEjemplos);

    float calcularError();
    float aplicarFuncionDeError(const float& actual, const float& prediccion);
    float aplicarFuncionDePromedioError(const float& error);

    void adjustarVariablesSegunEjemplo();
    void calcularGradienteSegunError(float& variable, float& deltaVariable);

    float OFFSET, multiplicadorDeEntrenamiento;

    int *forma;
    int numCapas, numWeights, numBias, numNodos;

    int *prevWeightsAtCapa, *prevBiasAtCapa, *prevNodosAtCapa;

    float *weights, *biases, *nodos;

    float *deltaWeights, *deltaBiases;

    float *objetivosActuales, errorPrediccionInicial;

    funcionesDeActivacion funcionDeActivacion;
    funcionesDeError funcionDeError;
};