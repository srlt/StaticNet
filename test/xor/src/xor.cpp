/**
 * @file   xor.cpp
 * @author Sébastien Rouault <sebmsg@free.fr>
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * any later version. Please see https://gnu.org/licenses/gpl.html
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * @section DESCRIPTION
 *
 * Trivial XOR test.
**/

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▁ Declarations ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// External headers
#include <iostream>
#include <cmath>

// Internal headers
#include <staticnet.hpp>

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

using namespace StaticNet;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Trivial XOR test ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Program entry point.
 * @param argc Ignored
 * @param argv Ignored
 * @return Always zero
**/
int main(int argc, char** argv) {
    Transfert sigmoid;
    if (!sigmoid.set(tanhf, -5, 5, 1000)) {
        std::cout << "Unable to set the transfert function" << std::endl;
        return 1;
    }

    Network<2, 2, 1> network(sigmoid);
    { // Initialization
        UniformRandomizer<std::ratio<1, 100>> randomizer;
        network.randomize(randomizer);
    }

    std::cout << "Raw network: ";
    network.print(std::cout);
    std::cout << std::endl;

    std::cout << std::endl;
    { // Training
        Learning<2, 1> discipline; // Learning discipline
        { // Discipline initialization
            Vector<2> input;
            Vector<1> output;

            discipline.add(input = {-1, -1}, output = {-1}, 0.1);
            discipline.add(input = {+1, -1}, output = {+1}, 0.1);
            discipline.add(input = {-1, +1}, output = {+1}, 0.1);
            discipline.add(input = {+1, +1}, output = {-1}, 0.1);
        }

        std::cout << "Learning discipline: ";
        discipline.print(std::cout);
        std::cout << std::endl;

        std::cout << "Learning...";
        std::cout.flush();
        std::cout << (discipline.correct(network, 0.1, 100000) ? " done." : " fail.") << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Network: ";
    network.print(std::cout);
    std::cout << std::endl;

    std::cout << std::endl << "Results: {" << std::endl;
    { // Test (on training set...)
        Vector<2> input;
        Vector<1> output;

        network.compute(input = {-1, -1}, output);
        std::cout << "\t";
        input.print(std::cout);
        std::cout << "\t-> ";
        output.print(std::cout);
        std::cout << std::endl;

        network.compute(input = {-1, +1}, output);
        std::cout << "\t";
        input.print(std::cout);
        std::cout << "\t-> ";
        output.print(std::cout);
        std::cout << std::endl;

        network.compute(input = {+1, -1}, output);
        std::cout << "\t";
        input.print(std::cout);
        std::cout << "\t-> ";
        output.print(std::cout);
        std::cout << std::endl;

        network.compute(input = {+1, +1}, output);
        std::cout << "\t";
        input.print(std::cout);
        std::cout << "\t-> ";
        output.print(std::cout);
        std::cout << std::endl;
    }
    std::cout << "}" << std::endl;

    return 0;
}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Trivial XOR test ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
