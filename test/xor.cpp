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
        val_t const eta = 0.01;
        val_t const err_limit = 0.001;
        val_t const iter_limit = 1000000;
        nat_t i = 0;
        while (true) { // Training
            val_t avgerr = 0; // Average square error
            Vector<1> expected;
            Vector<1> error;

            expected = {-0.7};
            network.correct({-1, -1}, expected, eta, sigmoid, error);
            avgerr += error.get(0) * error.get(0);

            expected = {+0.7};
            network.correct({-1, +1}, expected, eta, sigmoid, error);
            avgerr += error.get(0) * error.get(0);

            expected = {+0.7};
            network.correct({+1, -1}, expected, eta, sigmoid, error);
            avgerr += error.get(0) * error.get(0);

            expected = {-0.7};
            network.correct({+1, +1}, expected, eta, sigmoid, error);
            avgerr += error.get(0) * error.get(0);

            if (avgerr < err_limit || i > iter_limit) { // Arbitrary limit
                std::cout << "avgerr = " << avgerr / 4 << std::endl;
                network.print(std::cout);
                break;
            }
            i++;
        }
        std::cout << std::endl;
    }
    { // Test (on training set...)
        Vector<1> output;
        network.compute({-1, -1}, sigmoid, output);
        std::cout << "{-1, -1} -> " << output.get(0) << std::endl;
        network.compute({-1, +1}, sigmoid, output);
        std::cout << "{-1, +1} -> " << output.get(0) << std::endl;
        network.compute({+1, -1}, sigmoid, output);
        std::cout << "{+1, -1} -> " << output.get(0) << std::endl;
        network.compute({+1, +1}, sigmoid, output);
        std::cout << "{+1, +1} -> " << output.get(0) << std::endl;
    }

    return 0;
}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Trivial XOR test ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
