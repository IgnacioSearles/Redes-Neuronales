#include "superficieDeDibujo.h"

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

#include <iostream>
#include <algorithm>

superficieDeDibujo::superficieDeDibujo(sf::RenderWindow *mVentanaPrincipal, int mOffsetX, int mOffsetY, int mSizeX, int mSizeY)
{
    ventanaPrincipal = mVentanaPrincipal;

    offsetX = mOffsetX;
    offsetY = mOffsetY;
    sizeX = mSizeX;
    sizeY = mSizeY;

    grid.resize(28);
    for (int i = 0; i < 28; i++) grid.at(i).resize(28);
}

void superficieDeDibujo::actualizar() {
    dibujarGrid();
    procesarEventosMouse();
    dibujarCeldasGrid();
}

void superficieDeDibujo::dibujarGrid()
{
    for (int x = offsetX; x <= sizeX + offsetX; x += sizeX / 28)
    {
        sf::RectangleShape linea;
        linea.setSize(sf::Vector2f(1, sizeY));
        linea.setPosition(x, offsetY);
        ventanaPrincipal->draw(linea);
    }

    for (int y = offsetY; y <= sizeY + offsetY; y += sizeY / 28)
    {
        sf::RectangleShape linea;
        linea.setSize(sf::Vector2f(sizeX, 1));
        linea.setPosition(offsetX, y); 
        ventanaPrincipal->draw(linea);  
    }
}

void superficieDeDibujo::procesarEventosMouse() {
    sf::Vector2i posicionMouse = sf::Mouse::getPosition(*ventanaPrincipal);
    posicionMouse.x = (posicionMouse.x - offsetX) / (sizeX / 28);
    posicionMouse.y = (posicionMouse.y - offsetY) / (sizeY / 28);
    if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
        dibujarEnGrid(posicionMouse.x, posicionMouse.y);
    }
    else if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
        borrarEnGrid(posicionMouse.x, posicionMouse.y);
    }
}

void superficieDeDibujo::dibujarEnGrid(int xInGrid, int yInGrid) {
    for (float angulo = 0; angulo < 6.28f; angulo++) {
        for (float radio = 0; radio < 1.0f; radio += 0.5f) {
            int x = cosf(angulo) * radio + xInGrid;
            int y = sinf(angulo) * radio + yInGrid;
            if (y >= 0 && y < 28 && x >= 0 && x < 28) {
                float intensidad = (1.0f / (radio * 2));
                if (grid.at(y).at(x) + intensidad <= 1)
                    grid.at(y).at(x) += intensidad;
            }
        }
    }
}

void superficieDeDibujo::borrarEnGrid(int xInGrid, int yInGrid) {
    for (float angulo = 0; angulo < 6.28f; angulo++) {
        for (float radio = 0; radio < 1.0f; radio += 0.5f) {
            int x = cosf(angulo) * radio + xInGrid;
            int y = sinf(angulo) * radio + yInGrid;
            if (y >= 0 && y < 28 && x >= 0 && x < 28) {
                float intensidad = (1.0f / (radio * 2));
                if (grid.at(y).at(x) - intensidad >= 0)
                    grid.at(y).at(x) -= intensidad;
            }
        }
    }
}

void superficieDeDibujo::dibujarCeldasGrid() {
    for (int y = 0; y < grid.size(); y++) {
        for (int x = 0; x < grid.at(y).size(); x++) {
            sf::RectangleShape celda;
            celda.setSize(sf::Vector2f(sizeX / 28 - 2, sizeY / 28 - 2));
            celda.setPosition(x * (sizeX / 28) + offsetX + 1, y * (sizeY / 28) + offsetY + 1);
            celda.setFillColor(sf::Color(grid.at(y).at(x) * 255, grid.at(y).at(x) * 255, grid.at(y).at(x) * 255));
            ventanaPrincipal->draw(celda);
        }
    }
}

std::vector<float> superficieDeDibujo::obtenerPixeles() {
    cajaDeContencion caja = obtenerCajaDeContecionGrid();

    std::vector<float> salida;
    salida.resize(784);
    for (int i = 0; i < salida.size(); i++) salida.at(i) = 0;
    
    for (int y = caja.abajo; y < caja.arriba; y++) {
        for (int x = caja.izquierda; x < caja.derecha; x++) {
            float espacioVertical = (28.0f - (caja.arriba - caja.abajo)) / 2;
            float espacioHorizontal = (28.0f - (caja.derecha - caja.izquierda)) / 2;
            salida.at(((int)espacioVertical + y - caja.abajo) * 28 + ((int)espacioHorizontal + x - caja.izquierda)) = grid.at(y).at(x);
        }
    }

    return salida;
}

cajaDeContencion superficieDeDibujo::obtenerCajaDeContecionGrid() {
    cajaDeContencion caja(28, 0, 0, 28);

    for (int y = 0; y < grid.size(); y++) {
        for (int x = 0; x < grid.at(y).size(); x++) {
            if (grid.at(y).at(x) > 0) {
                if (x < caja.izquierda) caja.izquierda = x;
                if (x > caja.derecha) caja.derecha = x;
                if (y < caja.abajo) caja.abajo = y;
                if (y > caja.arriba) caja.arriba = y;
            }
        }
    }
    if (caja.izquierda == 28) return cajaDeContencion(0, 28, 28, 0);

    return caja;
}

void superficieDeDibujo::despejarGrid() {
    for (int y = 0; y < grid.size(); y++) {
        for (int x = 0; x < grid.at(y).size(); x++) {
            grid.at(y).at(x) = 0;
        }
    }
}