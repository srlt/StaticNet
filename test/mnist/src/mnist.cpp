/**
 * @file   mnist.cpp
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
 * MNIST database learning and recognition simple example.
**/

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▁ Declarations ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// External headers
#include <cstdio>
#include <stdexcept>
extern "C" {
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
}

// Internal headers
#include <staticnet.hpp>

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

using namespace StaticNet;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Constants ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// Constants
constexpr nat_t rows_length = 28; // Image row length
constexpr nat_t cols_length = 28; // Image col length
constexpr nat_t input_dim   = rows_length * cols_length; // Input space dimension
constexpr nat_t output_dim  = 10; // Output space dimension
val_t (*transfert_function)(val_t) = tanhf; // Transfert function used
val_t transfert_range = 2; // Difference between max and min
val_t default_margin = 0.1; // Default margin for the learning discipline

/** Input vector.
**/
using Input = Vector<input_dim>;

/** Output vector.
**/
using Output = Vector<output_dim>;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Constants ▔
// ▁ Simple transformations ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Return the digit associated with a dimension id.
 * @param dim Given dimension id
 * @return Associated digit
**/
constexpr nat_t output_dim_to_digit(nat_t dim) {
    return dim;
}

/** Return the dimension id associated with a digit.
 * @param digit Given digit
 * @return Associated dimension id
**/
constexpr nat_t output_digit_to_dim(nat_t digit) {
    return digit;
}

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

/** Initialize an output, and optionaly a margin vector from a label.
 * @param label  Label to translate
 * @param output Output vector
 * @param margin Margin vector (optional)
**/
void label_to_vector(nat_t label, Output& output, Output* margin = null) {
    for (nat_t i = 0; i < output_dim; i++)
        output.set(i, i == label ? 1 : -1);
    if (margin)
        for (nat_t i = 0; i < output_dim; i++)
            margin->set(i, i == label ? default_margin : transfert_range);
}

/** Translate an output vector to a label.
 * @param output Output vector to translate
 * @return Associated label
**/
nat_t vector_to_label(Output& output) {
    nat_t larger_dim = 0;
    nat_t larger_val = output.get(0);
    for (nat_t i = 1; i < output_dim; i++) {
        val_t val = output.get(i);
        if (val > larger_val) {
            larger_dim = i;
            larger_val = val;
        }
    }
    return output_dim_to_digit(larger_dim);
}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Simple transformations ▔
// ▁ Database ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Data parser from files.
**/
class Loader final {
private:
    /** File descriptor.
    **/
    using File = int;
    /** File map.
    **/
    class Map final {
    private:
        uint8_t* data;   // Data pages, null if none
        size_t   length; // Mapping length, in bytes
        size_t   cursor; // Cursor on the mapping
    public:
        /** Build a map to a file.
         * @param path Path to the file to map
        **/
        Map(char const* path) {
            File fd = open(path, O_RDONLY);
            if (fd == -1) {
                ::std::string err_str;
                err_str.append("Unable to open '");
                err_str.append(path);
                err_str.append("' for reading");
                throw ::std::runtime_error(err_str);
            }
            length = lseek(fd, 0, SEEK_END); // Get file size, in bytes
            data = static_cast<uint8_t*>(mmap(null, length, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
            if (data == MAP_FAILED)
                throw ::std::runtime_error("Mapping failed");
            close(fd);
            cursor = 0;
        }
        /** Destroy a map.
        **/
        ~Map() {
            if (likely(data))
                munmap(data, length);
        }
    public:
        /** Return a part of the data as an object of a given class, move the cursor to the next object.
         * @param Type Type of the returned object
         * @return Pointer to the object
        **/
        template<class Type> Type* read() {
            static_assert(!::std::is_polymorphic<Type>::value, "Given class is polymorphic");
            Type* ret = static_cast<Type*>(static_cast<void*>(data + cursor));
            cursor += sizeof(Type);
            if (cursor > length)
                throw ::std::runtime_error("Read out of file bounds");
            return ret;
        }
        /** Only move the cursor for a given number of bytes.
         * @param delta Number of bytes
        **/
        void seek(size_t delta) {
            cursor += delta;
        }
    };
    /** Image entry.
     * @param dim Entry total size
    **/
    class Entry final {
    private:
        uint8_t data[input_dim];
    private:
        /** Convert a grey-scale to an input level.
         * @param color Grey-scale to convert
         * @return Input level (-1 white ... +1 black)
        **/
        val_t convert(nat_t color) const {
            return static_cast<val_t>(color) / 255 * 2 - 1;
        }
    public:
        /** Initialize a vector with such data.
         * @param vector Vector to initialize
        **/
        void dump(Input& vector) const {
            for (nat_t i = 0; i < input_dim; i++)
                vector.set(i, convert(data[i]));
        }
    };
private:
    uint32_t const magic_img = 0x03080000; // Little-endian presumed
    uint32_t const magic_lab = 0x01080000; // <Same>
private:
    Map img; // File descriptor (-1 for none) for the images file
    Map lab; // File descriptor (-1 for none) for the labels file
    nat_t count;  // Remaining images
private:
    /** Inverse endianess.
     * @param UInt  Implicit unsigned integer type
     * @param value Value to inverse
     * @return Value inversed
    **/
    template<class Uint> Uint endian_inverse(Uint value) {
        Uint ret;
        uint8_t* v = static_cast<uint8_t*>(static_cast<void*>(&value));
        uint8_t* r = static_cast<uint8_t*>(static_cast<void*>(&ret));
        for (nat_t i = 0; i < sizeof(Uint); i++)
            r[i] = v[sizeof(Uint) - 1 - i];
        return ret;
    }
public:
    /** Open an images file, basic validity checks.
     * @param path_img Images file to open
     * @param path_lab Labels file to open
    **/
    Loader(char const* path_img, char const* path_lab): img(path_img), lab(path_lab) {
        uint32_t (&header_img)[4] = *img.read<uint32_t[4]>(); // Magic number, image count, row size, column size
        uint32_t (&header_lab)[2] = *lab.read<uint32_t[2]>(); // Magic number, label count
        if (header_img[2] != endian_inverse<uint32_t>(rows_length) || header_img[3] != endian_inverse<uint32_t>(cols_length)) {
            ::std::string err_str;
            err_str.append("'");
            err_str.append(path_img);
            err_str.append("' invalid dimensions");
            throw ::std::runtime_error(err_str);
        }
        if (header_img[1] != header_lab[1]) {
            ::std::string err_str;
            err_str.append("'");
            err_str.append(path_img);
            err_str.append("' and '");
            err_str.append(path_img);
            err_str.append("' count mismatch");
            throw ::std::runtime_error(err_str);
        }
        count = static_cast<nat_t>(endian_inverse(header_img[1]));
        if (count < 1) {
            ::std::string err_str;
            err_str.append("'");
            err_str.append(path_img);
            err_str.append("' and '");
            err_str.append(path_img);
            err_str.append("' no image");
            throw ::std::runtime_error(err_str);
        }
    }
public:
    /** Initialize an input vector and an associated label.
     * @param vector Input vector
     * @param label  Associated label
     * @return True if another vector/label exists, false otherwise
    **/
    bool feed(Input& vector, nat_t& label) {
        if (unlikely(count == 0))
            throw ::std::runtime_error("No more image to feed");
        Entry& image = *img.read<Entry>();
        uint8_t& lbl = *lab.read<uint8_t>();
        image.dump(vector);
        label = static_cast<nat_t>(lbl);
        return --count != 0;
    }
};

/** Tests set.
**/
class Tests final {
private:
    /** A labelled test image.
    **/
    class Image final {
        friend Tests; // Direct access for initialization
    private:
        Input image; // Associated input vector
        nat_t label; // Number represented
    public:
        /** Check if the network answered correctly.
         * @param network Network to test
         * @return True on a correct answer, false otherwise
        **/
        template<nat_t... implicit_dims> bool check(Network<implicit_dims...>& network) {
            Output result;
            network.compute(image, result);
            return vector_to_label(result) == label;
        }
    };
private:
    ::std::list<Image> tests; // List of test images with labels
public:
    /** Load images and labels from loader object.
     * @param loader Loader object to load from
    **/
    void load(Loader& loader) {
        while (true) { // At least one element in loader
            tests.emplace(tests.end());
            Image& current = tests.back(); // Current picture
            if (!loader.feed(current.image, current.label))
                break;
        }
    }
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

// Learning discipline used to train networks
Learning<input_dim, output_dim> discipline;

// Tests set
Tests tests;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Database ▔
// ▁ Entry point ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

/** Program entry point.
 * @param argc Ignored
 * @param argv Ignored
 * @return Always zero
**/
int main(int argc, char** argv) {
    { // Parameters handling
        if (argc != 5) { // Wrong number of parameters
            ::std::printf("Usage: %s <training images> <training labels> <test images> <test labels>\n", argc != 0 ? argv[0] : "mnist");
            return 0;
        }
        ::std::printf("Loading files...");
        fflush(stdout);
        try {
            Loader train(argv[1], argv[2]);
            Loader test(argv[3], argv[4]);
            { // Initialize learning discipline
                Input input;
                nat_t label;
                while (train.feed(input, label)) {
                    Output output;
                    Output margin;
                    label_to_vector(label, output, &margin);
                    discipline.add(input, output, margin);
                }
            }
            tests.load(test);
        } catch (::std::runtime_error& err) {
            ::std::printf(" fail: %s\n", err.what());
        }
        ::std::printf(" done.\n");
    }
    { // Learning phase
        ::std::printf("Learning phase...\n");
        /// TODO: Learning phase
    }
    { // Testing phase
        ::std::printf("Testing phase...\n");
        /// TODO: Testing phase
    }
    return 0;
}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Entry point ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
