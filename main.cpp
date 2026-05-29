#include <iostream>
#include <stdio.h>
#include "tensor.hpp"
#include "operation.hpp"
#include "mnist_reader.cpp"

int main() {
    std::string train_img = "mnist/train-images.idx3-ubyte";
    std::string train_lbl = "mnist/train-labels.idx1-ubyte";
    std::string test_img  = "mnist/t10k-images.idx3-ubyte";
    std::string test_lbl  = "mnist/t10k-labels.idx1-ubyte";

    std::cout << "Loading full dataset into memory..." << std::endl;
    MnistData full_data = load_full_mnist_dataset(train_img, train_lbl, test_img, test_lbl);

    std::cout << "\nSuccessfully loaded all data!" << std::endl;
    std::cout << "Train Images: " << full_data.num_train_images << std::endl;
    std::cout << "Test Images:  " << full_data.num_test_images << std::endl;
    std::cout << "Pixels/Image: " << full_data.pixels_per_image << std::endl;

    // Print an image from the TRAIN set (true)
    // print_mnist_image_float(full_data, true, 42); 

    ComputationContextHandle ctx = init_computation_context();
    const size_t batch_size = 1000;
    const size_t num_iterations = full_data.num_train_images / batch_size;
    const size_t input_size = full_data.pixels_per_image;
    const size_t num_labels = 10;

    const float beta = 0.1;
    const size_t num_epochs = 50;

    // inputs and outputs
    std::vector<std::pair<TensorHandle, TensorHandle>> data_slices;
    data_slices.reserve(num_iterations);
    for (size_t i = 0; i < num_iterations; ++i) {
        data_slices.push_back({
            tensor_from_data({batch_size, input_size}, false, full_data.images_train.data() + (i * batch_size * input_size)),
            tensor_from_data({batch_size, num_labels}, false, full_data.labels_train.data() + (i * batch_size * num_labels))
        });
    }

    // Learnable tensors
    TensorHandle w1 = tensor_random({input_size, 300}, true);
    TensorHandle b1 = tensor_random({300}, true);
    TensorHandle w2 = tensor_random({300, 100}, true);
    TensorHandle b2 = tensor_random({100}, true);
    TensorHandle w3 = tensor_random({100, num_labels}, true);
    TensorHandle b3 = tensor_random({num_labels}, true);

    std::vector<TensorHandle> learnable_tensors = {w1, b1, w2, b2, w3, b3};
    
    for (size_t epoch_ind = 0; epoch_ind < num_epochs; ++epoch_ind) {
        std::cout << "Epoch " << epoch_ind << ":" << std::endl;
        for (size_t it_ind = 0; it_ind < num_iterations; ++it_ind) {

            TensorHandle l2 = addition(
                mmul(data_slices[it_ind].first, w1, ctx), 
                broadcast(b1, batch_size, ctx), 
                ctx
            ); 
            TensorHandle z2 = relu(l2, ctx);

            TensorHandle l3 = addition(
                mmul(z2, w2, ctx), 
                broadcast(b2, batch_size, ctx), 
                ctx
            );
            TensorHandle z3 = relu(l3, ctx);

            TensorHandle l4 = addition(
                mmul(z3, w3, ctx), 
                broadcast(b3, batch_size, ctx), 
                ctx
            );
            TensorHandle z4 = relu(l4, ctx);

            TensorHandle p = softmax(z4, ctx);
            TensorHandle loss = cross_entropy(p, data_slices[it_ind].second, ctx);

            TensorHandle mean_loss = mean(loss, ctx);

            float final_loss = tensor_at(mean_loss, {});
            backward(mean_loss, ctx);
            optimize(learnable_tensors, beta);
            computation_context_clear(ctx);

            std::cout << "Iteration: " << it_ind << " Loss: " << final_loss << std::endl; 
        }
    }

    computation_context_free(ctx);
    return 0;
}