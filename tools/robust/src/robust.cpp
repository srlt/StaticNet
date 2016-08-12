/**
 * @file   robust.cpp
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
 * Compute the "robustness" of a feed-forward neural network, i.e.: bound maximal error on one output for a given amount of failed (returning 0) neurons.
 * Based on the paper "When Neurons Fail", El Mahdi El Mhamdi, Rachid Guerraoui, 2016.
**/

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▁ Declarations ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// External headers
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <vector>

// Internal headers
#include <staticnet.hpp>

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

using namespace StaticNet;

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

// Types used
using val_t = float;         // Any floating-point value
using nat_t = uint_fast32_t; // Any natural number

// Null value (I prefer null over nullptr)
auto const null = nullptr;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Robustness computation ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace Robust {

/** Spurious layer.
**/
class Layer final {
    friend class Network;
private:
    nat_t const input_dim;  // Input dimension
    nat_t const output_dim; // Output dimension
    nat_t const next_neurons; // Number of neurons on this layer and the next ones
    val_t max_weight; // Maximum weight amongst all neurons
public:
    /** Build a spurious layer for the given dimensions.
     * @param input_dim    Layer input dimension
     * @param output_dim   Layer output dimension
     * @param next_neurons Number of neurons on this layer and the next ones
    **/
    Layer(nat_t input_dim, nat_t output_dim, nat_t next_neurons): input_dim(input_dim), output_dim(output_dim), next_neurons(next_neurons) {}
private:
    /** Load layer weights, compute maximum weight.
     * @param input Input serializer
    **/
    void load(Serializer::Input& input) {
        val_t max = 0;
        for (nat_t i = 0; i < output_dim; i++) { // For each neuron of the current layer
            for (nat_t j = 0; j < input_dim; j++) { // For each weight of the current neuron
                val_t val = input.load();
                if (val < 0) // Absolute value
                    val = -val;
                if (val > max || i == 0) // Select maximum weight
                    max = val;
            }
            input.load(); // Discard bias
        }
        max_weight = max;
    }
};

/** Spurious feed-forward neural network.
**/
class Network final {
private:
    val_t lip; // Transfert function Lipschitz constant
    val_t cap; // Synapse transmission capacity
    nat_t nb_neurons; // Total number of neurons
    ::std::vector<Layer> layers; // Layers
public:
    /** Build a (spurious) network from a dimension string.
     * @param dim Dimensions string, null terminated
     * @param lip Transfert function Lipschitz constant
     * @param cap Synapse transmission capacity
    **/
    Network(char const* dim, val_t lip, val_t cap): lip(lip), cap(cap), nb_neurons(0) {
        nat_t size = 1;
        { // Get table size
            for (char const* cursor = dim;; cursor++) {
                char c = *cursor;
                if (c == '\0')
                    break;
                if (c == '-')
                    size++;
            }
            if (size >= 64) // Arbitrary limit
                throw ::std::runtime_error("Too many layers");
        }
        if (size < 2)
            throw ::std::runtime_error("At least one input and one output dimensions must be specified");
        nat_t dims[size];
        { // Split
            nat_t j = 0;
            nat_t v = 0;
            for (char const* cursor = dim;; cursor++) {
                char c = *cursor;
                if (c == '\0' || c == '-') {
                    dims[j++] = v;
                    if (c == '\0')
                        break;
                    v = 0;
                    continue;
                }
                if (c >= '0' && c <= '9') {
                    v = 10 * v + c - '0';
                    continue;
                }
                throw ::std::runtime_error("Invalid dimensions string");
            }
        }
        { // Layers creation
            for (nat_t i = 1; i < size; i++) // Sums up neuron counts
                nb_neurons += dims[i];
            nat_t next_neurons = nb_neurons;
            layers.reserve(size - 1);
            for (nat_t i = 1; i < size; i++) { // Initialize layers
                next_neurons -= dims[i];
                layers.push_back(Layer(dims[i - 1], dims[i], next_neurons));
            }
        }
    }
private:
    /** Compute the error bound of a layer.
     * @param layer    Given layer
     * @param err_fact Error factor
     * @param nb_byz   Number of byzantine neurons
     * @return Error bound value
    **/
    inline val_t layer_error(Layer const& layer, val_t err_fact, nat_t nb_byz) const {
        return static_cast<val_t>(layer.output_dim) + err_fact * static_cast<val_t>(layer.output_dim - nb_byz);
    }
    /** Compute the error bound from a given layer.
     * @param id       Layer id to compute the error from, starting at 1
     * @param nb_byz   Number of byzantine neurons, to spread across next layers
     * @param err_prev Error from the previous layer (0 for the first layer)
     * @return Maximum error from this layer for the given number of neurons
    **/
    val_t error(nat_t nb_byz, nat_t id = 1, val_t err_prev = val_t(0)) const {
        Layer const& layer = layers[id - 1];
        val_t const err_fact = err_prev * lip * layer.max_weight - val_t(1); // Error factor
        if (id == layers.size()) // Last layer
            return layer_error(layer, err_fact, nb_byz);
        { // Not last layer
            val_t max_err = 0;
            nat_t surplus;
            if (nb_byz > layer.output_dim) { // More byzantine neurons than neurons on the layer
                surplus = nb_byz - layer.output_dim;
                nb_byz = layer.output_dim;
            } else {
                surplus = 0;
            }
            for (nat_t i = 0; i <= nb_byz; i++) { // Number of neurons transmitted to the next layers (without the surplus)
                if (i + surplus <= layer.next_neurons) { // Enough neurons in the next layers
                    val_t err = layer_error(layer, err_fact, nb_byz - i);
                    err = error(i + surplus, id + 1, err);
                    if (err > max_err) // Keep the maximal error
                        max_err = err;
                }
            }
            return max_err;
        }
    }
public:
    /** Load network weights.
     * @param input Input serializer
    **/
    void load(Serializer::Input&& input) {
        for (Layer& layer: layers)
            layer.load(input);
    }
    /** Output max error points to a stream.
     * @param out Out stream
    **/
    void output(::std::ostream& out) const {
        if (unlikely(nb_neurons == nat_t(-1))) // Loops would be infinite
            throw ::std::runtime_error("Too many byzantine neurons");
        for (nat_t i = 0; i < nb_neurons; i++)
            out << i + 1 << "\t" << error(i + 1) * cap << ::std::endl;
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Robustness computation ▔
// ▁ Entry point ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Program entry point.
 * @param argc Number of arguments
 * @param argv Arguments
 * @return Return code
**/
int main(int argc, char** argv) {
    if (argc != 4) {
        ::std::cerr << "Usage: 'network' | " << (argc < 1 ? "robust" : argv[0]) << " <dimensions> <transfert absolute maximum> <transfert Lipschitz constant> | 'max error/failed neurons data points'" << ::std::endl;
        return 0;
    }
    Robust::Network network(argv[1], ::std::stof(argv[2]), ::std::stof(argv[3])); // Spurious network
    network.load(Serializer::StreamInput(::std::cin)); // Load a StaticNet network
    network.output(::std::cout); // Output data points
    return 0;
}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Entry point ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
