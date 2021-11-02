#pragma once

#include <SFML/Graphics.hpp>
#include <vector>

struct cajaDeContencion {
    int izquierda, derecha, arriba, abajo;
    cajaDeContencion(int mIz, int mDer, int mArr, int mAba) {
        izquierda = mIz;
        derecha = mDer;
        arriba = mArr;
        abajo = mAba;
    }
};

class superficieDeDibujo {
public:
    superficieDeDibujo(sf::RenderWindow* mVentanaPrincipal, int mOffsetX, int mOffsetY, int mSizeX, int mSizeY);
    void actualizar();
    void despejarGrid();
    std::vector<float> obtenerPixeles();
private:
    cajaDeContencion obtenerCajaDeContecionGrid();
    void procesarEventosMouse();
    void dibujarGrid();

    void dibujarEnGrid(int xInGrid, int yInGrid);
    void borrarEnGrid(int xInGrid, int yInGrid);

    void dibujarCeldasGrid();

    int offsetX, offsetY, sizeX, sizeY;

    std::vector<std::vector<float>> grid;
    sf::RenderWindow* ventanaPrincipal;
};