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

void extract_statistics(const std::vector<std::chrono::duration<double>>& times_vector, double metrics[][4], int test_index) {
    double total_time = 0.0;
    double min_time = std::numeric_limits<double>::max(); // Initialize min_time to a large value
    double max_time = std::numeric_limits<double>::min(); // Initialize max_time to a small value

    for (const auto& time : times_vector) {
        double time_sec = time.count(); // Convert duration to seconds
        
        // Update total_time
        total_time += time_sec;

        // Update min_time and max_time
        if (time_sec < min_time) min_time = time_sec;
        if (time_sec > max_time) max_time = time_sec;
    }
    double average_time = total_time / times_vector.size();

    metrics[test_index][0] = total_time;
    metrics[test_index][1] = average_time;
    metrics[test_index][2] = min_time;
    metrics[test_index][3] = max_time;
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

    // Creating an array for storring the time data
    std::vector<std::chrono::duration<double>> load_image_time_data;
    std::vector<std::chrono::duration<double>> pre_process_time_data;
    std::vector<std::chrono::duration<double>> cnn_run_time_data;

    // Starging a timer for measuring total execution time
    auto total_exec_time_start = std::chrono::high_resolution_clock::now();

    // Running with a specific batch size
    for (unsigned int n = 0; n < images.size(); n += batchSize) {
        unsigned int runSize = (images.size() < (n + batchSize)) ? (images.size() - n) : batchSize;
        in_dims[0] = runSize;
        out_dims[0] = batchSize;

        for (unsigned int i = 0; i < runSize; i++) {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Image load starts here
            // Starging a timer and loading the image to the RAM
            auto load_image_time_start = std::chrono::high_resolution_clock::now();
            Mat image = imread(baseImagePath + images[n + i]);
            auto load_image_time_end = std::chrono::high_resolution_clock::now();

            // Calculating the time needed to load the image and appending them to the vector
            std::chrono::duration<double> load_image_time = load_image_time_end - load_image_time_start;
            load_image_time_data.push_back(load_image_time);
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Image load ends here

            // Updating the image row and columns
            input_height = image.rows;
            input_width = image.cols;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Image pre-process starts here
            // Starting a timer and performing image pre-process
            auto pre_process_time_start = std::chrono::high_resolution_clock::now();
            Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
            resize(image, image2, Size(inHeight, inWidth), 0, 0, INTER_NEAREST);

            for (int h = 0; h < inHeight; h++) {
                for (int w = 0; w < inWidth; w++) {
                    for (int c = 0; c < 3; c++)
                        imageInputs[i*inSize+h*inWidth*3+w*3 +2-c] = (int8_t)( ( (image2.at<Vec3b>(h, w)[c]/255.0f - mean[2-c] ) /std_dev[2-c] ) *input_scale );
                }
            }
            auto pre_process_time_end = std::chrono::high_resolution_clock::now();

            // Calculating the time needed to pre-process the image and appending them to the vector
            std::chrono::duration<double> pre_process_time = pre_process_time_end - pre_process_time_start;
            pre_process_time_data.push_back(pre_process_time);
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Image pre-process ends here
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

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Running the model starts here
        // Running on the model
        auto cnn_run_time_start = std::chrono::high_resolution_clock::now();
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);
        auto cnn_run_time_end = std::chrono::high_resolution_clock::now();

        // Calculating the time needed to run the cnn and appending them to the vector
        std::chrono::duration<double> cnn_run_time = cnn_run_time_end - cnn_run_time_start;
        cnn_run_time_data.push_back(cnn_run_time);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Running the model ends here
        for (unsigned int i = 0; i < runSize; i++) {
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

    double metrics[3][4] = {0};
    extract_statistics(load_image_time_data, metrics, 0);
    extract_statistics(pre_process_time_data, metrics, 1);
    extract_statistics(cnn_run_time_data, metrics, 2);
    
    // Define the file path
    std::string filename = "../../time_measurements.txt";

    // Open the file
    FILE* filePtr = fopen(filename.c_str(), "a");
    if (filePtr != nullptr) {
        fprintf(filePtr, "***********************************************************************************************************\n");
        fprintf(filePtr, "Test for %dx%d\n", input_height, input_width);
        fprintf(filePtr, "Total number of images: %d\n\n", images.size());

        fprintf(filePtr, "Load image time:\n");
        fprintf(filePtr, "Total time: %f, Average time: %f, Min time: %f, Max time: %f\n\n", metrics[0][0], metrics[0][1], metrics[0][2], metrics[0][3]);

        fprintf(filePtr, "Pre-process time:\n");
        fprintf(filePtr, "Total time: %f, Average time: %f, Min time: %f, Max time: %f\n\n", metrics[1][0], metrics[1][1], metrics[1][2], metrics[1][3]);

        fprintf(filePtr, "CNN run time:\n");
        fprintf(filePtr, "Total time: %f, Average time: %f, Min time: %f, Max time: %f\n\n", metrics[2][0], metrics[2][1], metrics[2][2], metrics[2][3]);

        fprintf(filePtr, "Total execution time: %f\n", total_exec_time.count());
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
