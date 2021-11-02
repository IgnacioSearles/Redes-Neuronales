#include "boton.h"
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>

boton::boton(sf::RenderWindow* mVentanaPrincipal, sf::Font* mTipografia, const std::string& mTextoBoton, int mX, int mY) {
    ventanaPrincipal = mVentanaPrincipal;
    tipografia = mTipografia;
    textoBoton = mTextoBoton;
    x = mX;
    y = mY;

    xSize = 10;
    ySize = 10;
}

void boton::actualizar() {
    sf::RectangleShape cajaBoton;
    cajaBoton.setPosition(x - 5, y - 5);
    cajaBoton.setSize(sf::Vector2f(xSize, ySize));
    cajaBoton.setFillColor(sf::Color(180, 180, 180));
    cajaBoton.setOutlineThickness(1);
    cajaBoton.setOutlineColor(sf::Color::White);
    ventanaPrincipal->draw(cajaBoton);

    sf::Text texto;
    texto.setFont(*tipografia);
    texto.setString(textoBoton);
    texto.setPosition(x, y);
    texto.setCharacterSize(20);
    texto.setFillColor(sf::Color::Black);
    ventanaPrincipal->draw(texto);

    sf::FloatRect cajaTexto = texto.getLocalBounds();
    xSize = cajaTexto.width + 10;
    ySize = cajaTexto.height + 10;
}

bool boton::estaCliqueado() {
    sf::Vector2i posicionMouse = sf::Mouse::getPosition(*ventanaPrincipal);

    if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && posicionMouse.x >= x - 5 && posicionMouse.x < x - 5 + xSize && posicionMouse.y >= y - 5 && posicionMouse.y < y - 5 + ySize)
        return true;
    
    return false;
}