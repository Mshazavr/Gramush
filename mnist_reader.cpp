// Clanker code for loading input dataset

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

// ---------------------------------------------------------
// 1. Data Structure Definition
// ---------------------------------------------------------
struct MnistData {
    uint32_t num_train_images;
    uint32_t num_test_images;
    uint32_t pixels_per_image; // 28 * 28 = 784

    std::vector<float> images_train; // Normalized [0.0, 1.0]
    std::vector<float> labels_train; // 1-hot encoded (size: num_train_images * 10)
    
    std::vector<float> images_test;  // Normalized [0.0, 1.0]
    std::vector<float> labels_test;  // 1-hot encoded (size: num_test_images * 10)
};

// Helper function to swap endianness for Intel/AMD CPUs
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
           ((val <<  8) & 0x00FF0000) |
           ((val >>  8) & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

// ---------------------------------------------------------
// 2. Private Helper Functions for Reading Files
// ---------------------------------------------------------
std::vector<float> read_mnist_images(const std::string& path, uint32_t& out_num_images, uint32_t& out_pixels_per_image) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open image file: " + path);
    }

    uint32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic = swap_endian(magic);
    out_num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != 2051) throw std::runtime_error("Invalid magic number in image file: " + path);

    out_pixels_per_image = rows * cols;
    uint32_t total_pixels = out_num_images * out_pixels_per_image;
    
    std::vector<uint8_t> temp_raw(total_pixels);
    file.read(reinterpret_cast<char*>(temp_raw.data()), total_pixels);
    
    std::vector<float> float_images(total_pixels);
    for (uint32_t i = 0; i < total_pixels; ++i) {
        float_images[i] = static_cast<float>(temp_raw[i]) / 255.0f;
    }

    return float_images;
}

std::vector<float> read_mnist_labels(const std::string& path, uint32_t expected_num_images) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + path);
    }

    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic = swap_endian(magic);
    num_labels = swap_endian(num_labels);

    if (magic != 2049) throw std::runtime_error("Invalid magic number in label file: " + path);
    if (num_labels != expected_num_images) throw std::runtime_error("Label count mismatch in file: " + path);

    std::vector<uint8_t> temp_raw(num_labels);
    file.read(reinterpret_cast<char*>(temp_raw.data()), num_labels);
    
    // 1-hot encode
    std::vector<float> float_labels(num_labels * 10, 0.0f);
    for (uint32_t i = 0; i < num_labels; ++i) {
        uint8_t digit = temp_raw[i];
        if (digit < 10) {
            float_labels[(i * 10) + digit] = 1.0f;
        }
    }

    return float_labels;
}

// ---------------------------------------------------------
// 3. The Main Loading Function
// ---------------------------------------------------------
MnistData load_full_mnist_dataset(
    const std::string& train_img_path, const std::string& train_lbl_path,
    const std::string& test_img_path, const std::string& test_lbl_path) 
{
    MnistData dataset;

    // Load Training Set
    dataset.images_train = read_mnist_images(train_img_path, dataset.num_train_images, dataset.pixels_per_image);
    dataset.labels_train = read_mnist_labels(train_lbl_path, dataset.num_train_images);

    // Load Testing Set
    uint32_t test_pixels;
    dataset.images_test = read_mnist_images(test_img_path, dataset.num_test_images, test_pixels);
    
    if (test_pixels != dataset.pixels_per_image) {
        throw std::runtime_error("Train and Test sets have different image dimensions!");
    }
    
    dataset.labels_test = read_mnist_labels(test_lbl_path, dataset.num_test_images);

    return dataset;
}

// ---------------------------------------------------------
// 4. Print Helper Function
// ---------------------------------------------------------
void print_mnist_image_float(const MnistData& dataset, bool from_train_set, uint32_t image_index) {
    uint32_t max_index = from_train_set ? dataset.num_train_images : dataset.num_test_images;
    if (image_index >= max_index) {
        std::cerr << "Index out of bounds!" << std::endl;
        return;
    }

    // Create const references to point to whichever vectors we are trying to read from
    const std::vector<float>& target_images = from_train_set ? dataset.images_train : dataset.images_test;
    const std::vector<float>& target_labels = from_train_set ? dataset.labels_train : dataset.labels_test;

    int actual_digit = -1;
    uint32_t label_offset = image_index * 10;
    for (int i = 0; i < 10; ++i) {
        if (target_labels[label_offset + i] == 1.0f) {
            actual_digit = i;
            break;
        }
    }

    std::string set_name = from_train_set ? "TRAIN" : "TEST";
    std::cout << "\nDisplaying " << set_name << " image index: " << image_index 
              << " | Decoded Digit Label: " << actual_digit << "\n";
    std::cout << "1-Hot Array: [";
    for (int i = 0; i < 10; ++i) {
        std::cout << target_labels[label_offset + i] << (i < 9 ? ", " : "");
    }
    std::cout << "]\n------------------------------------------\n";

    uint32_t img_offset = image_index * dataset.pixels_per_image;
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            if (target_images[img_offset + (r * 28) + c] > 0.5f) {
                std::cout << "██"; 
            } else {
                std::cout << "  "; 
            }
        }
        std::cout << "\n";
    }
    std::cout << "------------------------------------------\n";
}
