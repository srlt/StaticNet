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
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
}

// Internal headers
#include <staticnet.hpp>

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

using namespace StaticNet;

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Declarations ▔
// ▁ Database ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

// Constants
constexpr nat_t rows_length = 28; // Image row length
constexpr nat_t cols_length = 28; // Image col length
constexpr nat_t input_dim   = rows_length * cols_length; // Input space dimension
constexpr nat_t output_dim  = 10; // Output space dimension

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

/** Input vector.
**/
using Input = Vector<input_dim>;

/** Output vector.
**/
using Output = Vector<output_dim>;

/** Tests set.
**/
class Tests final {
private:
    /** A labelled test image.
    **/
    class Image final {
    private:
        Vector<input_dim> image; // Associated input vector
        nat_t label; // Number represented
    public:
        /** Check if the network answered correctly.
         * @param network Network to test
         * @return True on a correct answer, false otherwise
        **/
        template<nat_t... implicit_dims> bool check(Network<implicit_dims...>& network) {
            Output result;
            network.compute(image, result);
            { // "Larger" dimension as result
                nat_t larger_dim = 0;
                nat_t larger_val = result.get(0);
                for (nat_t i = 1; i < output_dim; i++) {
                    val_t val = result.get(i);
                    if (val > larger_val) {
                        larger_dim = i;
                        larger_val = val;
                    }
                }
                return output_dim_to_digit(larger_dim) == label;
            }
        }
    };
private:
    ::std::list<Image> tests; // List of test images
public:

};

/** Data parser from files.
**/
class Loader final {
private:
    /** File descriptor.
    **/
    using File = int;
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
        void dump(Vector<input_dim>& vector) const {
            for (nat_t i = 0; i < input_dim; i++)
                vector.set(i, convert(data[i]));
        }
    };
private:
    constexpr static uint8_t magic_img[] = { 0x00, 0x00, 0x08, 0x01 }; // Expected magic number for an images file
    constexpr static uint8_t magic_lab[] = { 0x00, 0x00, 0x08, 0x03 }; // Expected magic number for a labels file
private:
    File  fd_img; // File descriptor (-1 for none) for the images file
    File  fd_lab; // File descriptor (-1 for none) for the labels file
    nat_t count; // Remaining images
private:
    /** Open file for reading, throw on error.
     * @param path Path to file
     * @param fd   File descriptor (output)
    **/
    void open_file(char const* path, File& fd) {
        fd = open(path, O_RDONLY);
        if (fd == -1) {
            ::std::string err_str;
            err_str.append("Unable to open '");
            err_str.append(path);
            err_str.append("' for reading");
            throw ::std::runtime_error(err_str);
        }
    }
    /** Read file, throw on error.
     * @param fd     File descriptor
     * @param buffer Buffer to write
     * @param size   Size to write
     * @param path   Path to file (error message)
    **/
    void read_file(File& fd, char* data, int size, char const* path) {
        if (read(fd, data, size) != size) {
            ::std::string err_str;
            err_str.append("Unable to read '");
            err_str.append(path);
            err_str.append("'");
            throw ::std::runtime_error(err_str);
        }
    }
public:
    /** Open an images file, basic validity checks.
     * @param path_img Images file to open
     * @param path_lab Labels file to open
    **/
    Loader(char const* path_img, char const* path_lab): fd_img(-1), fd_lab(-1), count() {
        { // Open files
            open_file(path_img, fd_img);
            open_file(path_lab, fd_lab);
        }
        uint32_t header_img[4]; // Magic number, image count, row size, column size
        uint32_t header_lab[2]; // Magic number, label count
        { // Basic checks
            read_file(fd_img, reinterpret_cast<char*>(header_img), sizeof(uint32_t) * 4, path_img);
            read_file(fd_lab, reinterpret_cast<char*>(header_lab), sizeof(uint32_t) * 2, path_lab);
            if (memcmp(&header_img[0], magic_img, sizeof(uint32_t))) {
                ::std::string err_str;
                err_str.append("'");
                err_str.append(path_img);
                err_str.append("' is not an images file");
                throw ::std::runtime_error(err_str);
            }
            if (memcmp(&header_lab[0], magic_lab, sizeof(uint32_t))) {
                ::std::string err_str;
                err_str.append("'");
                err_str.append(path_lab);
                err_str.append("' is not an labels file");
                throw ::std::runtime_error(err_str);
            }
            if (header_img[2] != rows_length || header_img[3] != cols_length) {
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
            count = static_cast<nat_t>(header_img[1]);
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
    }
    /** Close the file descriptors, if open.
    **/
    ~Loader() {
        if (fd_img != -1)
            close(fd_img);
        if (fd_lab != -1)
            close(fd_lab);
    }
public:
    /** Initialize an input vector and an associated label.
     * @param vector Input vector
     * @param label  Associated label
     * @return True if another vector/label exists, false otherwise
    **/
    bool feed(Vector<input_dim>& vector, nat_t& label) {
        if (unlikely(count == 0))
            throw ::std::runtime_error("No more image to feed");
        /// TODO: Feed input vector/label
        return --count != 0;
    }
};

// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

// Learning discipline used to train networks
Learning<input_dim, output_dim> discipline;



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
        /// TODO: Parameters handling
    }
    return 0;
}

// ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔ Entry point ▔
// ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
