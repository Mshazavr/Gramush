#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct MnistData {
  uint32_t num_train_images;
  uint32_t num_test_images;
  uint32_t pixels_per_image;

  std::vector<float> images_train;
  std::vector<float> labels_train;

  std::vector<float> images_test;
  std::vector<float> labels_test;
};

MnistData load_full_mnist_dataset(const std::string &train_img_path,
                                  const std::string &train_lbl_path,
                                  const std::string &test_img_path,
                                  const std::string &test_lbl_path);

void print_mnist_image_float(const MnistData &dataset, bool from_train_set,
                             uint32_t image_index);
