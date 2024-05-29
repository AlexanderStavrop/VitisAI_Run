/*

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023
*/

// based on  Vitis AI 3.0 VART "resnet50.cc" demo code

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <fstream>  // Include for file operations

#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

GraphInfo shapes;

//const string baseImagePath = "./test/";
//const string wordsPath = "./";
string baseImagePath, wordsPath;  // they will get their values via argv[]

/**
* @brief put image names to a vector
*
* @param path - path of the image direcotry
* @param images - the vector of image name
*
* @return none
*/
void ListImages(string const& path, vector<string>& images) {
    images.clear();
    struct dirent* entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
                (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
            images.push_back(name);
            }
        }
    }
    closedir(dir);
}

/**
* @brief load kinds from file to a vector
*
* @param path - path of the kinds file
* @param kinds - the vector of kinds string
*
* @return none
*/
void LoadWords(string const& path, vector<string>& kinds) {
    kinds.clear();
    ifstream fkinds(path);
    if (fkinds.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }
    string kind;
    while (getline(fkinds, kind))
        kinds.push_back(kind);

    fkinds.close();
}

/**
* @brief calculate softmax
*
* @param data - pointer to input buffer
* @param size - size of input buffer
* @param result - calculation result
*
* @return none
*/
void CPUCalcSoftmax(const int8_t* data, size_t size, float* result, float scale) {
  assert(data && result);
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp((float)data[i] * scale);
    sum += result[i];
  }
  for (size_t i = 0; i < size; i++)
    result[i] /= sum;
}

void ArgMax(const int8_t* data,  size_t size, float *res_val, int *res_index, float scale) {
    int index = 0;
    int8_t max = data[0];
    for (size_t i = 1; i < size; i++) {
        if (data[i] > max) {
            max = data[i];
            index = i;
        }
    }
    *res_val   = (float) (max * scale);
    *res_index = index;
}

/**
* @brief Get top k results according to its probability
*
* @param d - pointer to input data
* @param size - size of input data
* @param k - calculation result
* @param vkinds - vector of kinds
*
* @return none
*/
void TopK(const float* d, int size, int k, vector<string>& vkinds) {
  assert(d && size > 0 && k > 0);
  priority_queue<pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(pair<float, int>(d[i], i));
  }

  for (auto i = 0; i < k; ++i) {
    pair<float, int> ki = q.top();
    q.pop();
  }
}


/**
* @brief Run DPU Task for CNN
*
* @return none
*/
void run_CNN(vart::Runner* runner) {
    // Creating the needed vector
    vector<string> kinds, images;

    // Load all image names.
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: No images existing under " << baseImagePath << endl;
        return;
    }

    // Load all kinds words.
    LoadWords(wordsPath, kinds);
    if (kinds.size() == 0) {
        cerr << "\nError: No words exist in file " << wordsPath << endl;
        return;
    }

    //B G R format
    //R_MEAN = 123.68
    //G_MEAN = 116.78
    //B_MEAN = 103.94
    //
    // R 125.20 G 122.91 B 113.73
    //Channels = [B_MEAN,G_MEAN, R_MEAN]
    // To get the corresponding normalization numbers, do <norm_in_1.0_range> * 255.0
    float mean[3] = {0.4914, 0.4822, 0.4465};
    float std_dev[3] = {0.2023, 0.1994, 0.2010};

    // Getting in/out tensors and dims
    auto outputTensors = runner->get_output_tensors();
    auto inputTensors = runner->get_input_tensors();
    auto out_dims = outputTensors[0]->get_shape();
    auto in_dims = inputTensors[0]->get_shape();

    auto input_scale = get_input_scale(inputTensors[0]);
    auto output_scale = get_output_scale(outputTensors[0]);

    // Getting shape info
    int outSize = shapes.outTensorList[0].size;
    int inSize = shapes.inTensorList[0].size;
    int inHeight = shapes.inTensorList[0].height;
    int inWidth = shapes.inTensorList[0].width;
    int batchSize = in_dims[0];

    // Creating a vector for inputs and outputs
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

    // Setting the image variables
    vector<Mat> imageList;
    int8_t* imageInputs = new int8_t[inSize * batchSize];

    float* softmax = new float[outSize];
    int8_t* FCResult = new int8_t[batchSize * outSize];
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

    // MIGHT NOT BE NEEDED
    int input_height, input_width;

    // Creating an array for storring the time needed to load every image
    std::vector<std::chrono::duration<double>> load_image_time_data;

    // Starging a timer for measuring total execution time
    auto total_exec_time_start = std::chrono::high_resolution_clock::now();

    // Running with a specific batch size
    for (unsigned int n = 0; n < images.size(); n += batchSize) {
        unsigned int runSize = (images.size() < (n + batchSize)) ? (images.size() - n) : batchSize;
        in_dims[0] = runSize;
        out_dims[0] = batchSize;

        for (unsigned int i = 0; i < runSize; i++) {
            // Starging a timer for measuring image load time
            auto load_image_time_start = std::chrono::high_resolution_clock::now();

            // Loading the image on RAM
            Mat image = imread(baseImagePath + images[n + i]);

            // Ending the timer for measuring image load time
            auto load_image_time_end = std::chrono::high_resolution_clock::now();

            // Calculating the time needed to load the image and appending them to the vector
            load_image_time_data.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(load_image_time_end - load_image_time_start));

            // for debug
//             cout << "Original Image Dimensions - Width: " << image.cols << ", Height: " << image.rows << ", Channels: " << image.channels() << endl;
            input_height = image.rows;
            input_width = image.cols;

            // Performing image pre-process
            Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
            resize(image, image2, Size(inHeight, inWidth), 0, 0, INTER_NEAREST);
            // For debug
//             printf("inHeight = %d\ninWidth = %d\n", inHeight, inWidth);

            for (int h = 0; h < inHeight; h++) {
                for (int w = 0; w < inWidth; w++) {
                    for (int c = 0; c < 3; c++)
                        imageInputs[i*inSize+h*inWidth*3+w*3 +2-c] = (int8_t)( ( (image2.at<Vec3b>(h, w)[c]/255.0f - mean[2-c] ) /std_dev[2-c] ) *input_scale );
                }
            }
        imageList.push_back(image);
        }

        // in/out tensor refactory for batch inout/output
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(inputTensors[0]->get_name(), in_dims, xir::DataType{xir::DataType::XINT, 8u})));
        inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(imageInputs, batchTensors.back().get()));
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(outputTensors[0]->get_name(), out_dims, xir::DataType{xir::DataType::XINT, 8u})));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(FCResult, batchTensors.back().get()));

        // tensor buffer input/output
        inputsPtr.clear();
        outputsPtr.clear();
        inputsPtr.push_back(inputs[0].get());
        outputsPtr.push_back(outputs[0].get());

        // Running on the model
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);
        for (unsigned int i = 0; i < runSize; i++) {
            // For debug
//             cout << "\nImage : " << images[n + i] << endl;

            // Calculating the softmax on CPU and display TOP-5 classification results
            CPUCalcSoftmax(&FCResult[i * outSize], outSize, softmax,  output_scale);
            TopK(softmax, outSize, 5, kinds);
        }

        imageList.clear();
        inputs.clear();
        outputs.clear();
    }
    auto total_exec_time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_exec_time = total_exec_time_end - total_exec_time_start;

    double total_time = 0.0;
    double min_time = std::numeric_limits<double>::max(); // Initialize min_time to a large value
    double max_time = std::numeric_limits<double>::min(); // Initialize max_time to a small value

    for (const auto& time : load_image_time_data) {
        double time_sec = time.count(); // Convert duration to seconds
        total_time += time_sec;

        // Update min_time and max_time
        if (time_sec < min_time) {
            min_time = time_sec;
        }
        if (time_sec > max_time) {
            max_time = time_sec;
        }
    }
    double average_time = total_time / load_image_time_data.size();

    // Define the file path
    std::string filename = "../../time_measurements.txt";

    // Open the file
    FILE* filePtr = fopen(filename.c_str(), "a");
    if (filePtr != nullptr) {
        fprintf(filePtr, "Test for %dx%d\n", input_height, input_width);
        // Write to the file using fprintf
        fprintf(filePtr, "Average load image time %.6f seconds (With min: %.6f seconds, Max: %.6f seconds)\n", average_time, min_time, max_time);
        fprintf(filePtr, "Total execution time: %.6f seconds\n", total_exec_time.count());
        fprintf(filePtr, "***********************************************************************************************************\n\n");

        // Close the file
        fclose(filePtr);
    } else {
        cerr << "Error: Unable to open file for writing" << endl;
    }

    delete[] FCResult;
    delete[] imageInputs;
    delete[] softmax;
}

/**
 * @brief Entry for running CNN
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy CNN on DPU platform.
 *
 */
int main(int argc, char* argv[]){
    // Check args
    if (argc != 4) {
        cout << "Usage: <executable> <xmodel> <test_images_dir>, <labels_filename>" << endl;
        return -1;
    }

    baseImagePath = std::string(argv[2]); //path name of the folder with test images
    wordsPath     = std::string(argv[3]); //filename of the labels

    auto graph = xir::Graph::deserialize(argv[1]);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(), 1u)
        << "CNN should have one and only one dpu subgraph.";
    LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
    /*create runner*/
    auto runner = vart::Runner::create_runner(subgraph[0], "run");
    // ai::XdpuRunner* runner = new ai::XdpuRunner("./");
    /*get in/out tensor*/
    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();

    /*get in/out tensor shape*/
    int inputCnt = inputTensors.size();
    int outputCnt = outputTensors.size();
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes;
    getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

    /*run with batch*/
    run_CNN(runner.get());
    return 0;
}
