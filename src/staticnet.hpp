/**
 * @file   staticnet.hpp
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
 * StaticNet: a toy feed-forward neural network implementation with compile-time-known configuration.
**/

#ifndef SRC_STATICNET_HPP
#define SRC_STATICNET_HPP
// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▁ Declarations ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// External headers
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <random>

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

namespace StaticNet {

// Types used
typedef float         val_t; // Any floating-point value
typedef uint_fast32_t nat_t; // Any natural number

}

// Null value
#undef  null
#define null nullptr

/** Specify a proposition as 'likely true'.
 * @param prop Proposition likely true
**/
#undef likely
#ifdef __GNUC__
    #define likely(prop) \
        __builtin_expect((prop) ? true : false, true)
#else
    #define likely(prop) \
        (prop)
#endif

/** Specify a proposition as 'likely false'.
 * @param prop Proposition likely false
**/
#undef unlikely
#ifdef __GNUC__
    #define unlikely(prop) \
        __builtin_expect((prop) ? true : false, false)
#else
    #define unlikely(prop) \
        (prop)
#endif

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Random number generator ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Abstract randomizer.
**/
class Randomizer {
public:
    /** Get a random number.
    **/
    virtual val_t get() = 0;
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Uniform distribution randomizer.
**/
class UniformRandomizer final: public Randomizer {
private:
    std::random_device                    device;
    std::default_random_engine            engine;
    std::uniform_real_distribution<val_t> distrib;
public:
    /** Constructor.
    **/
    UniformRandomizer(): device(), engine(device()), distrib(-0.01, 0.01) {
    }
public:
    /** Get a random number.
    **/
    val_t get() {
        return distrib(engine);
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Random number generator ▔
// ▁ Transfert function ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Transfert function.
**/
class Transfert final {
private:
    constexpr static val_t diff_delta = 0.0001; // Delta for derivative estimation
private:
    nat_t  count; // Nb points
    val_t  x_min; // Min input
    val_t  x_max; // Max input
    val_t  delta; // dx between points
    val_t* tbase; // Points of base function (null if not initialized)
    val_t* tdiff; // Points of derived function (= base + prec)
private:
    /** Get a value through this function/its derivative, perform linear interpolation.
     * @param func Selected function ('base' or 'diff')
     * @param x    Input value
     * @return Output value
    **/
    enum class select { base, diff }; // Function type selector
    template<select func> val_t get(val_t x) const {
        if (unlikely(x < x_min)) {
            return (func == select::diff ? 0 : tbase[0]);
        } else if (unlikely(x >= x_max)) {
            return (func == select::diff ? 0 : tbase[count - 1]);
        } // Else linear interpolation
        nat_t i = static_cast<nat_t>((x - x_min) / delta);
        if (unlikely(i + 1 >= count)) { // Due to floating-point imprecision
            return tbase[count - 1];
        } else {
            val_t f = (x - (x_min + static_cast<val_t>(i) * delta)) / delta;
            val_t y_a = (func == select::diff ? tdiff : tbase)[i];
            val_t y_b = (func == select::diff ? tdiff : tbase)[i + 1];
            return y_a + (y_b - y_a) * f;
        }
    }
public:
    /** Constructor.
    **/
    Transfert(): tbase(null) {
    }
    /** Destructor.
    **/
    ~Transfert() {
        if (tbase)
            ::free(static_cast<void*>(tbase));
    }
public:
    /** Get a value through this function/its derivative, perform linear interpolation.
     * @param x Input value
     * @return Output value
    **/
    val_t operator()(val_t x) const {
        return get<select::base>(x);
    }
    val_t diff(val_t x) const {
        return get<select::diff>(x);
    }
    /** (Re)set the transfert function.
     * @param trans Transfert function
     * @param min   Min input
     * @param max   Max input
     * @param prec  Amount of points
     * @return True if the operation is a success, false otherwise
    **/
    bool set(val_t trans(val_t), val_t min, val_t max, nat_t prec) {
        if (unlikely(min >= max || prec < 2)) // Basic checks
            return false;
        { // Points table allocation
            if (tbase) // Points table freeing (if already exists)
                ::free(tbase);
            void* addr = ::malloc(2 * prec * sizeof(val_t)); // Both tables
            if (!addr) { // Allocation failure
                tbase = null;
                return false;
            }
            tbase = static_cast<val_t*>(addr);
            tdiff = tbase + prec;
        }
        { // Tables initialization
            nat_t i;
            delta = (max - min) / static_cast<val_t>(prec - 1);
            i = 0;
            for (val_t x = min; i < prec; x += delta) // Base
                tbase[i++] = trans(x);
            i = 0;
            for (val_t x = min; i < prec; x += delta) // Diff
                tdiff[i++] = (trans(x + diff_delta / 2) - trans(x - diff_delta / 2)) / diff_delta;
        }
        { // Basic finalization
            count = prec;
            x_min = min;
            x_max = max;
        }
        return true;
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Transfert function ▔
// ▁ Simple vector ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Simple vector class.
 * @param dim Vector dimension
**/
template<nat_t dim> class Vector final {
    static_assert(dim > 0, "Invalid vector dimension");
private:
    val_t vec[dim]; // Vector
public:
    /** Uninitialized constructor.
    **/
    Vector() {
    }
    /** Copy constructor.
     * @param copy Vector to copy
    **/
    Vector(Vector<dim> const& copy) {
        for (nat_t i = 0; i < dim; i++)
            vec[i] = copy.get(i);
    }
    /** Initializer list constructor.
     * @param params Initial values (cardinality must be the dimension of the vector)
    **/
    Vector(std::initializer_list<val_t> params) {
        /// FIXME: Non-constant condition here. Missing a 'correction' brought by C++17 ? See 'http://en.cppreference.com/w/cpp/utility/initializer_list/size' then 'http://en.cppreference.com/w/cpp/iterator/distance'.
        // static_assert(params.size() != dim, "Wrong initializer list size");
        nat_t i = 0;
        for (val_t v: params)
            vec[i++ % dim] = v; /// NOTE: Temporary modulo
    }
public:
    /** Get a single coordinate.
     * @param id Coordinate id
     * @return Coordinate value
    **/
    val_t get(nat_t id) const {
        return vec[id];
    }
    /** Set a single coordinate.
     * @param id Coordinate id
     * @param cv Coordinate value
    **/
    void set(nat_t id, val_t cv) {
        vec[id] = cv;
    }
public:
    /** Copy assignment.
     * @param x Vector to copy
     * @return Current vector
    **/
    Vector<dim>& operator=(Vector<dim> const& x) {
        for (nat_t i = 0; i < dim; i++)
            set(i, x.get(i));
        return *this;
    }
    /** Initializer list assignment.
     * @param params Initial values (cardinality must be the dimension of the vector)
     * @return Current vector
    **/
    Vector<dim>& operator=(std::initializer_list<val_t> params) {
        /// FIXME: Non-constant condition here. Same remark.
        // static_assert(params.size() != dim, "Wrong initializer list size");
        nat_t i = 0;
        for (val_t v: params)
            vec[i++ % dim] = v; /// NOTE: Temporary modulo
        return *this;
    }
    /** Scalar product.
     * @param x Vector to substract
     * @return Current vector
    **/
    val_t operator*(Vector<dim> const& x) const {
        val_t sum = 0;
        for (nat_t i = 0; i < dim; i++)
            sum += get(i) * x.get(i);
        return sum;
    }
public:
    /** Load vector data from dump.
     * @param dump Dump to read
     * @return New dump address
    **/
    val_t* load(val_t* dump) {
        for (nat_t i = 0; i < dim; i++)
            set(i, dump[i]);
        return dump + dim * sizeof(val_t);
    }
    /** Store vector data to dump.
     * @param dump Dump to write
     * @return New dump address
    **/
    val_t* store(val_t* dump) {
        for (nat_t i = 0; i < dim; i++)
            dump[i] = get(i);
        return dump + dim * sizeof(val_t);
    }
    /** Print vector to the given stream.
     * @param ostr Output stream
    **/
    void print(std::ostream& ostr) {
        ostr << "{ " << get(0);
        for (nat_t i = 1; i < dim; i++)
            ostr << ", " << get(i);
        ostr << " }";
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Simple vector ▔
// ▁ Neural Network ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Single neuron with synapses.
 * @param input_dim Input vector dimension
**/
template<nat_t input_dim> class Neuron final {
    static_assert(input_dim > 0, "Invalid input vector dimension");
public:
    Vector<input_dim> weight; // Input weight vector
    val_t bias; // Bias
public:
    /** Randomize the weight vector.
     * @param rand Randomizer to use
    **/
    void randomize(Randomizer& rand) {
        for (nat_t i = 0; i < input_dim; i++)
            weight.set(i, rand.get());
        bias = rand.get();
    }
    /** Compute the output of the neuron.
     * @param input   Input vector
     * @param trans   Transfert function
     * @param out_sum Sum of weighted inputs (optional)
     * @return Output scalar
    **/
    val_t compute(Vector<input_dim> const& input, Transfert const& trans, val_t* out_sum = null) {
        val_t sum = weight * input + bias;
        if (out_sum)
            *out_sum = sum;
        return trans(sum);
    }
    /** Correct the weight vector of the neuron.
     * @param input Input vector
     * @param sum   Sum of weighted inputs
     * @param error Sum of weighted errors
     * @param eta   Correction factor
     * @param trans Transfert function
     * @return Error scalar
    **/
    val_t correct(Vector<input_dim> const& input, val_t sum, val_t error, val_t eta, Transfert const& trans) {
        val_t err = error * trans.diff(sum);
        for (nat_t i = 0; i < input_dim; i++)
            weight.set(i, weight.get(i) + eta * err * input.get(i));
        bias += eta * err;
        return err;
    }
public:
    /** Load neuron data from dump.
     * @param dump Dump to read
     * @return New dump address
    **/
    val_t* load(val_t* dump) {
        dump = weight.load(dump);
        bias = *dump;
        return dump + 1;
    }
    /** Store neuron data to dump.
     * @param dump Dump to write
     * @return New dump address
    **/
    val_t* store(val_t* dump) {
        dump = weight.store(dump);
        *dump = bias;
        return dump + 1;
    }
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(std::ostream& ostr) {
        ostr << "{ ";
        weight.print(ostr);
        ostr << ", " << bias << " }";
    }
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Layer of neurons.
 * @param input_dim  Input vector dimension
 * @param output_dim Output vector dimension
**/
template<nat_t input_dim, nat_t output_dim> class Layer final {
    static_assert(input_dim > 0, "Invalid input vector dimension");
    static_assert(output_dim > 0, "Invalid output vector dimension");
private:
    Neuron<input_dim> neurons[output_dim]; // Neurons
public:
    /** Randomize the layer.
     * @param rand Randomizer to use
    **/
    void randomize(Randomizer& rand) {
        for (nat_t i = 0; i < output_dim; i++)
            neurons[i].randomize(rand);
    }
    /** Compute the output vector of the layer.
     * @param input   Input vector
     * @param trans   Transfert function
     * @param output  Output vector
     * @param out_sum Sum of weighted inputs vector (output, optional)
    **/
    void compute(Vector<input_dim> const& input, Transfert const& trans, Vector<output_dim>& output, Vector<output_dim>* out_sum = null) {
        if (out_sum) {
            for (nat_t i = 0; i < output_dim; i++) {
                val_t sum;
                output.set(i, neurons[i].compute(input, trans, &sum));
                out_sum->set(i, sum);
            }
        } else {
            for (nat_t i = 0; i < output_dim; i++)
                output.set(i, neurons[i].compute(input, trans));
        }
    }
    /** Correct the neurons of the layer.
     * @param input     Input vector
     * @param sums      Sum of weighted inputs vector
     * @param error     Sum of weighted errors vector
     * @param eta       Correction factor
     * @param trans     Transfert function
     * @param error_out Sum of weighted errors vector (optional)
    **/
    void correct(Vector<input_dim> const& input, Vector<output_dim> const& sums, Vector<output_dim> const& error, val_t eta, Transfert const& trans, Vector<input_dim>* error_out = null) {
        if (error_out) { // Error vector asked
            Vector<output_dim> errors; // Neuron errors
            for (nat_t i = 0; i < output_dim; i++)
                errors.set(i, neurons[i].correct(input, sums.get(i), error.get(i), eta, trans));
            for (nat_t i = 0; i < input_dim; i++) { // Compute error vector
                val_t sum = 0; // Sum of weighted error
                for (nat_t j = 0; j < output_dim; j++)
                    sum += neurons[j].weight.get(i) * errors.get(j);
                error_out->set(i, sum);
            }
        } else {
            for (nat_t i = 0; i < output_dim; i++)
                neurons[i].correct(input, sums.get(i), error.get(i), eta, trans);
        }
    }
public:
    /** Load layer data from dump.
     * @param dump Dump to read
     * @return New dump address
    **/
    val_t* load(val_t* dump) {
        for (nat_t i = 0; i < output_dim; i++)
            dump = neurons[i].load(dump);
        return dump;
    }
    /** Store layer data to dump.
     * @param dump Dump to write
     * @return New dump address
    **/
    val_t* store(val_t* dump) {
        for (nat_t i = 0; i < output_dim; i++)
            dump = neurons[i].store(dump);
        return dump;
    }
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(std::ostream& ostr) {
        ostr << "{" << std::endl << "\t";
        neurons[0].print(ostr);
        for (nat_t i = 1; i < output_dim; i++) {
            ostr << "," << std::endl << "\t";
            neurons[i].print(ostr);
        }
        ostr << std::endl << "}";
    }
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Network of layers, right folded.
 * @param ... Input/output vector dimensions
**/
template<nat_t input_dim, nat_t inter_dim, nat_t... output_dim> class Network final {
    static_assert(input_dim > 0, "Invalid input vector dimension");
    static_assert(inter_dim > 0, "Invalid intermediate vector dimension");
private:
    Layer<input_dim, inter_dim>       layer;  // Input layer
    Network<inter_dim, output_dim...> layers; // Output network
public:
    /** Randomize the network.
     * @param rand Randomizer to use
    **/
    void randomize(Randomizer& rand) {
        layer.randomize(rand);
        layers.randomize(rand);
    }
    /** Compute the output vector of the network.
     * @param input  Input vector
     * @param trans  Transfert function
     * @param output Output vector
    **/
    template<nat_t implicit_dim> void compute(Vector<input_dim> const& input, Transfert const& trans, Vector<implicit_dim>& output) {
        Vector<inter_dim> local_output; // Local layer output vector
        layer.compute(input, trans, local_output);
        layers.compute(local_output, trans, output);
    }
    /** Compute then reduce the quadratic error of the network.
     * @param input     Input vector
     * @param expected  Expected output vector
     * @param eta       Correction factor
     * @param trans     Transfert function
     * @param error     Error vector (output)
     * @param error_out <Reserved>
    **/
    template<nat_t implicit_dim> void correct(Vector<input_dim> const& input, Vector<implicit_dim> const& expected, val_t eta, Transfert const& trans, Vector<implicit_dim>& error, Vector<input_dim>* error_out = null) {
        Vector<inter_dim> local_output;
        Vector<inter_dim> local_sums;
        layer.compute(input, trans, local_output, &local_sums);
        Vector<inter_dim> local_error;
        layers.correct(local_output, expected, eta, trans, error, &local_error);
        layer.correct(input, local_sums, local_error, eta, trans, error_out);
    }
public:
    /** Load layer data from dump.
     * @param dump Dump to read
     * @return New dump address
    **/
    val_t* load(val_t* dump) {
        return layers.load(layer.load(dump));
    }
    /** Store layer data to dump.
     * @param dump Dump to write
     * @return New dump address
    **/
    val_t* store(val_t* dump) {
        return layers.store(layer.store(dump));
    }
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(std::ostream& ostr) {
        layer.print(ostr);
        ostr << ", ";
        layers.print(ostr);
    }
};
template<nat_t input_dim, nat_t output_dim> class Network<input_dim, output_dim> final {
    static_assert(input_dim > 0, "Invalid input vector dimension");
    static_assert(output_dim > 0, "Invalid output vector dimension");
private:
    Layer<input_dim, output_dim> layer; // Input/output layer
public:
    /** Randomize the network.
     * @param rand Randomizer to use
    **/
    void randomize(Randomizer& rand) {
        layer.randomize(rand);
    }
    /** Compute the output vector of the network.
     * @param input  Input vector
     * @param trans  Transfert function
     * @param output Output vector
    **/
    void compute(Vector<input_dim> const& input, Transfert const& trans, Vector<output_dim>& output) {
        layer.compute(input, trans, output);
    }
    /** Compute then reduce the quadratic error of the network.
     * @param input    Input vector
     * @param expected Expected output vector
     * @param eta      Correction factor
     * @param trans    Transfert function
     * @param error    Error vector (output)
     * @param error_out <Reserved>
    **/
    void correct(Vector<input_dim> const& input, Vector<output_dim> const& expected, val_t eta, Transfert const& trans, Vector<output_dim>& error, Vector<input_dim>* error_out = null) {
        Vector<output_dim> local_output;
        Vector<output_dim> local_sums;
        layer.compute(input, trans, local_output, &local_sums);
        for (nat_t i = 0; i < output_dim; i++)
            error.set(i, expected.get(i) - local_output.get(i));
        layer.correct(input, local_sums, error, eta, trans, error_out);
    }
public:
    /** Load layer data from dump.
     * @param dump Dump to read
     * @return New dump address
    **/
    val_t* load(val_t* dump) {
        return layer.load(dump);
    }
    /** Store layer data to dump.
     * @param dump Dump to write
     * @return New dump address
    **/
    val_t* store(val_t* dump) {
        return layer.store(dump);
    }
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(std::ostream& ostr) {
        layer.print(ostr);
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Neural Network ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
#endif
