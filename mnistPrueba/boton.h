#pragma once

#include <SFML/Graphics.hpp>

class boton {
public:
    boton(sf::RenderWindow* mVentanaPrincipal, sf::Font* mTipografia, const std::string& mTextoBoton, int mX, int mY);
    void actualizar();
    bool estaCliqueado();
private:
    sf::RenderWindow *ventanaPrincipal;
    sf::Font *tipografia;
    std::string textoBoton;
    int x, y, xSize, ySize;
};