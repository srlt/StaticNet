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
 * A toy, header-only feed-forward neural network implementation with compile-time-known neural network dimensions.
**/

#ifndef SRC_STATICNET_HPP
#define SRC_STATICNET_HPP
// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▁ Declarations ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// External headers
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <random>
#include <ratio>

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

namespace StaticNet {

// Types used
using val_t = float;         // Any floating-point value
using nat_t = uint_fast32_t; // Any natural number

// Null value (I prefer null over nullptr)
auto const null = nullptr;

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

}

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
 * @param Ratio Uniform distribution over [-ratio(), +ratio()]
**/
template<class Ratio> class UniformRandomizer final: public Randomizer {
private:
    /** Compute floating-point ratio from the ::std::ratio type "compatible" template parameter.
     * @return Floating-point representation of the ratio
    **/
    constexpr val_t ratio() {
        constexpr val_t ret = (val_t) Ratio::num / (val_t) Ratio::den;
        static_assert(ret > 0, "'Ratio' must be a positive value");
        return ret;
    }
private:
    ::std::random_device                    device;
    ::std::default_random_engine            engine;
    ::std::uniform_real_distribution<val_t> distrib;
public:
    /** Constructor.
    **/
    UniformRandomizer(): device(), engine(device()), distrib(-ratio(), ratio()) {}
public:
    /** Get a random number.
     * @return A random number
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
            return (func == select::diff ? tdiff : tbase)[0];
        } else if (unlikely(x >= x_max)) {
            return (func == select::diff ? tdiff : tbase)[count - 1];
        } // Else linear interpolation
        nat_t i = static_cast<nat_t>((x - x_min) / delta);
        if (unlikely(i + 1 >= count)) { // Due to floating-point imprecision
            return (func == select::diff ? tdiff : tbase)[count - 1];
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
    Transfert(): tbase(null) {}
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
                ::free(static_cast<void*>(tbase));
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
// ▁ Input/Output serializer ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {
namespace Serializer {

/** Abstract input serializer class.
**/
class Input {
public:
    /** Load one value, in order of writing.
     * @return Value loaded
    **/
    virtual val_t load() = 0;
};

/** Abstract output serializer class.
**/
class Output {
public:
    /** Store one value.
     * @param Value stored
    **/
    virtual void store(val_t) = 0;
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Input serializer based on a stream.
**/
class StreamInput final: public Input {
private:
    ::std::istream& istream; // Input stream
public:
    /** Build a simple input stream.
     * @param istream Input stream to use
    **/
    StreamInput(::std::istream& istream): istream(istream) {}
public:
    /** Load one value.
     * @return value Value stored
    **/
    val_t load() {
        val_t value;
        istream.read(reinterpret_cast<::std::istream::char_type*>(&value), sizeof(val_t));
        return value;
    }
};

/** Output serializer based on a stream.
**/
class StreamOutput final: public Output {
private:
    ::std::ostream& ostream; // Output stream
public:
    /** Build a simple output stream.
     * @param ostream Output stream to use
    **/
    StreamOutput(::std::ostream& ostream): ostream(ostream) {}
public:
    /** Store one value.
     * @param value Value stored
    **/
    void store(val_t value) {
        ostream.write(reinterpret_cast<::std::istream::char_type*>(&value), sizeof(val_t));
    }
};

} }

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Input/Output serializer ▔
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
    Vector() {}
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
    Vector(::std::initializer_list<val_t> params) {
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
    Vector<dim>& operator=(::std::initializer_list<val_t> params) {
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
    /** Vector comparison.
     * @param x Vector to compare
     * @return True if equal, false otherwise
    **/
    bool operator==(Vector<dim> const& x) const {
        for (nat_t i = 0; i < dim; i++)
            if (x.get(i) != get(i))
                return false;
        return true;
    }
public:
    /** Return the size of the structure.
     * @return Size of the structure, in bytes
    **/
    constexpr size_t size() {
        return dim * sizeof(val_t);
    }
    /** Load vector data.
     * @param input Serialized input
    **/
    void load(Serializer::Input& input) {
        for (nat_t i = 0; i < dim; i++)
            set(i, input.load());
    }
    /** Store vector data.
     * @param output Serialized output
    **/
    void store(Serializer::Output& output) {
        for (nat_t i = 0; i < dim; i++)
            output.store(get(i));
    }
public:
    /** Print vector to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) {
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
    /** Return the size of the structure.
     * @return Size of the structure, in bytes
    **/
    constexpr size_t size() {
        return weight.size() + sizeof(val_t);
    }
    /** Load neuron data.
     * @param input Serialized input
    **/
    void load(Serializer::Input& input) {
        weight.load(input);
        bias = input.load();
    }
    /** Store vector data.
     * @param output Serialized output
    **/
    void store(Serializer::Output& output) {
        weight.store(output);
        output.store(bias);
    }
public:
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) {
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
    Transfert const& trans; // Transfert function to use
    Neuron<input_dim> neurons[output_dim]; // Neurons
public:
    /** Layer constructor.
     * @param trans Transfert function to use
    **/
    Layer(Transfert const& trans): trans(trans) {}
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
     * @param output  Output vector
     * @param out_sum Sum of weighted inputs vector (output, optional)
    **/
    void compute(Vector<input_dim> const& input, Vector<output_dim>& output, Vector<output_dim>* out_sum = null) {
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
     * @param error_out Sum of weighted errors vector (optional)
    **/
    void correct(Vector<input_dim> const& input, Vector<output_dim> const& sums, Vector<output_dim> const& error, val_t eta, Vector<input_dim>* error_out = null) {
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
    /** Return the size of the structure.
     * @return Size of the structure, in bytes
    **/
    constexpr size_t size() {
        size_t size = 0;
        for (nat_t i = 0; i < output_dim; i++)
            size += neurons[i].size();
        return size;
    }
    /** Load layer data.
     * @param input Serialized input
    **/
    void load(Serializer::Input& input) {
        for (nat_t i = 0; i < output_dim; i++)
            neurons[i].load(input);
    }
    /** Store vector data.
     * @param output Serialized output
    **/
    void store(Serializer::Output& output) {
        for (nat_t i = 0; i < output_dim; i++)
            neurons[i].store(output);
    }
public:
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) {
        ostr << "{" << ::std::endl << "\t";
        neurons[0].print(ostr);
        for (nat_t i = 1; i < output_dim; i++) {
            ostr << "," << ::std::endl << "\t";
            neurons[i].print(ostr);
        }
        ostr << ::std::endl << "}";
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
    /** Network constructor.
     * @param trans Transfert function to use
    **/
    Network(Transfert const& trans): layer(trans), layers(trans) {}
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
     * @param output Output vector
    **/
    template<nat_t implicit_dim> void compute(Vector<input_dim> const& input, Vector<implicit_dim>& output) {
        Vector<inter_dim> local_output; // Local layer output vector
        layer.compute(input, local_output);
        layers.compute(local_output, output);
    }
    /** Compute then reduce the quadratic error of the network.
     * @param input     Input vector
     * @param expected  Expected output vector
     * @param eta       Correction factor
     * @param error     Error vector (output)
     * @param error_out <Reserved>
    **/
    template<nat_t implicit_dim> void correct(Vector<input_dim> const& input, Vector<implicit_dim> const& expected, val_t eta, Vector<implicit_dim>& error, Vector<input_dim>* error_out = null) {
        Vector<inter_dim> local_output;
        Vector<inter_dim> local_sums;
        layer.compute(input, local_output, &local_sums);
        Vector<inter_dim> local_error;
        layers.correct(local_output, expected, eta, error, &local_error);
        layer.correct(input, local_sums, local_error, eta, error_out);
    }
public:
    /** Return the size of the structure.
     * @return Size of the structure, in bytes
    **/
    constexpr size_t size() {
        return layer.size() + layers.size();
    }
    /** Load layer data.
     * @param input Serialized input
    **/
    void load(Serializer::Input& input) {
        layer.load(input);
        layers.load(input);
    }
    /** Store vector data.
     * @param output Serialized output
    **/
    void store(Serializer::Output& output) {
        layer.store(output);
        layers.store(output);
    }
public:
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) {
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
    /** Network constructor.
     * @param trans Transfert function to use
    **/
    Network(Transfert const& trans): layer(trans) {}
public:
    /** Randomize the network.
     * @param rand Randomizer to use
    **/
    void randomize(Randomizer& rand) {
        layer.randomize(rand);
    }
    /** Compute the output vector of the network.
     * @param input  Input vector
     * @param output Output vector
    **/
    void compute(Vector<input_dim> const& input, Vector<output_dim>& output) {
        layer.compute(input, output);
    }
    /** Compute then reduce the quadratic error of the network.
     * @param input     Input vector
     * @param expected  Expected output vector
     * @param eta       Correction factor
     * @param error     Error vector (output)
     * @param error_out <Reserved>
    **/
    void correct(Vector<input_dim> const& input, Vector<output_dim> const& expected, val_t eta, Vector<output_dim>& error, Vector<input_dim>* error_out = null) {
        Vector<output_dim> local_output;
        Vector<output_dim> local_sums;
        layer.compute(input, local_output, &local_sums);
        for (nat_t i = 0; i < output_dim; i++)
            error.set(i, expected.get(i) - local_output.get(i));
        layer.correct(input, local_sums, error, eta, error_out);
    }
public:
    /** Return the size of the structure.
     * @return Size of the structure, in bytes
    **/
    constexpr size_t size() {
        return layer.size();
    }
    /** Load layer data.
     * @param input Serialized input
    **/
    void load(Serializer::Input& input) {
        layer.load(input);
    }
    /** Store vector data.
     * @param output Serialized output
    **/
    void store(Serializer::Output& output) {
        layer.store(output);
    }
public:
    /** Print neuron weights to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) {
        layer.print(ostr);
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Neural Network ▔
// ▁ Learning discipline ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

namespace StaticNet {

/** Learning discipline.
 * @param input_dim  Input vector dimensions
 * @param output_dim Output vector dimensions
**/
template<nat_t input_dim, nat_t output_dim> class Learning final {
    static_assert(input_dim > 0, "Invalid input vector dimension");
    static_assert(output_dim > 0, "Invalid output vector dimension");
private:
    /** Input vector type.
    **/
    using Input  = Vector<input_dim>;
    /** Output vector type.
    **/
    using Output = Vector<output_dim>;
    /** Input, and expected output, with error margin.
    **/
    class Constraint final {
    private:
        Input  input;    // Input vector
        Output expected; // Expected output vector
        Output margin;   // Tolerated margin
    public:
        /** Build a new constraint.
         * @param input    Input vector
         * @param expected Expected output vector
         * @param margin   Tolerated margin vector
        **/
        Constraint(Input& input, Output& expected, Output& margin): input(input), expected(expected), margin(margin) {}
    public:
        /** Check equality between input vectors.
         * @param input Input vector to compare with
         * @return True if the stored input vector is equal to the given one, false otherwise
        **/
        bool match(Input& input) {
            return this->input == input;
        }
        /** Correct the network one time, if needed.
         * @param network Neural network to correct
         * @param eta     Correction factor
         * @return True if on bounds, false if a correction has been applied
        **/
        template<nat_t... implicit_dims> bool correct(Network<implicit_dims...>& network, val_t eta) {
            Output output; // Output vector
            network.compute(input, output);
            for (nat_t i = 0; i < output_dim; i++) { // Check for bounds
                val_t diff = expected.get(i) - output.get(i);
                if ((diff < 0 ? -diff : diff) > margin.get(i)) { // Out of at least one bound
                    network.correct(input, expected, eta, output);
                    return false;
                }
            }
            return true;
        }
    public:
        /** Print constraint to the given stream.
         * @param ostr Output stream
        **/
        void print(::std::ostream& ostr) {
            ostr << "{ ";
            input.print(ostr);
            ostr << ", ";
            expected.print(ostr);
            ostr << ", ";
            margin.print(ostr);
            ostr << " }";
        }
    };
private:
    ::std::vector<Constraint> constraints; // Constraints set
    ::std::random_device device; // Random device
    ::std::default_random_engine engine; // Default engine
public:
    /** Build an empty learning discipline.
    **/
    Learning(): constraints(), device(), engine(device()) {}
public:
    /** Add a constraint to the discipline, not checked for duplicate.
     * @param input  Input vector
     * @param output Expected output vector
     * @param margin Tolerated margin vector
    **/
    void add(Input& input, Output& output, Output& margin) {
        constraints.emplace(constraints.end(), input, output, margin);
    }
    /** Tell if a constraint exists based on the input vector.
     * @param input Input vector of the constraint to find
     * @return True if a matching constraint has been found, false otherwise
    **/
    bool has(Input& input) {
        for (Constraint& constraint: constraints)
            if (constraint.match(input))
                return true;
        return false;
    }
    /** Remove a constraint based on the input vector.
     * @param input Input vector of the constraint to remove
    **/
    void remove(Input& input) {
        nat_t pos = 0;
        for (Constraint& constraint: constraints) {
            if (constraint.match(input)) {
                constraints.erase(pos);
                return;
            }
            pos++;
        }
    }
    /** Remove all constraints.
    **/
    void reset() {
        constraints.clear();
    }
public:
    /** Correct the network one time, so that each output is near enough from its expected output.
     * @param network Neural network to correct
     * @param eta     Correction factor
     * @return Number of out-bounds constraints
    **/
    template<nat_t... implicit_dims> nat_t correct(Network<implicit_dims...>& network, val_t eta) {
        nat_t count = 0;
        for (Constraint& constraint: constraints) {
            if (!constraint.correct(network, eta)) // Not in-bounds
                count++;
        }
        return count;
    }
    /** Randomize constraints order.
    **/
    void shuffle() {
        ::std::shuffle(constraints.begin(), constraints.end(), engine);
    }
public:
    /** Print learning discipline to the given stream.
     * @param ostr Output stream
    **/
    void print(::std::ostream& ostr) {
        if (constraints.empty()) {
            ostr << "{}";
            return;
        }
        ostr << "{" << ::std::endl << "\t";
        bool first = true;
        for (Constraint& constraint: constraints) {
            if (first) {
                first = false;
                constraint.print(ostr);
            } else {
                ostr << "," << ::std::endl << "\t";
                constraint.print(ostr);
            }
        }
        ostr << ::std::endl << "}";
    }
};

}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Learning discipline ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
#endif
