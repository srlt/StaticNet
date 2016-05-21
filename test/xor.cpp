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
    using namespace StaticNet;

    Transfert sigmoid;
    if (!sigmoid.set([](val_t x) { return tanhf(x); }, -5, 5, 1000)) {
        std::cout << "Unable to set the transfert function" << std::endl;
        return 1;
    }

    Network<2, 2, 1> network;
    { // Initialization
        UniformRandomizer randomizer;
        network.randomize(randomizer);
    }
    { // Training
        val_t const eta = 0.05;
        val_t const err_limit = 0.001;
        val_t const iter_limit = 1000000;

        nat_t nb_iter = 0;
        while (true) { // (Bad) training
            val_t avg_err = 0; // Average square error distance
            Vector<1> expected;
            Vector<1> error;

            expected = {-0.7};
            network.correct({-1, -1}, expected, eta, sigmoid, error);
            avg_err += error * error;

            expected = {+0.7};
            network.correct({-1, +1}, expected, eta, sigmoid, error);
            avg_err += error * error;

            expected = {+0.7};
            network.correct({+1, -1}, expected, eta, sigmoid, error);
            avg_err += error * error;

            expected = {-0.7};
            network.correct({+1, +1}, expected, eta, sigmoid, error);
            avg_err += error * error;

            avg_err /= 4;
            if (avg_err < err_limit || nb_iter >= iter_limit) {
                std::cout << "nb_iter = " << nb_iter << std::endl
                          << "avg_err = " << avg_err << std::endl;
                network.print(std::cout);
                break;
            }
            nb_iter++;
        }
        std::cout << std::endl;
    }
    { // Test (on training set...)
        Vector<2> input;
        Vector<1> output;

        network.compute(input = {-1, -1}, sigmoid, output);
        input.print(std::cout);
        std::cout << "\t-> ";
        output.print(std::cout);
        std::cout << std::endl;

        network.compute(input = {-1, +1}, sigmoid, output);
        input.print(std::cout);
        std::cout << "\t-> ";
        output.print(std::cout);
        std::cout << std::endl;

        network.compute(input = {+1, -1}, sigmoid, output);
        input.print(std::cout);
        std::cout << "\t-> ";
        output.print(std::cout);
        std::cout << std::endl;

        network.compute(input = {+1, +1}, sigmoid, output);
        input.print(std::cout);
        std::cout << "\t-> ";
        output.print(std::cout);
        std::cout << std::endl;
    }

    return 0;
}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Trivial XOR test ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
