# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming

While both face detection and grayscale conversion can be performed with CUDA, they represent very different levels of complexity.

# Grayscale Conversion: 
This is a relatively simple pixel-wise operation and is a great introductory example for CUDA.
Face Detection: This is a much more complex task that typically involves machine learning models (like Haar cascades or deep learning models like CNNs). Implementing a full-fledged face detection algorithm from scratch using CUDA would be a monumental undertaking, far beyond a typical code example.
Therefore, I will provide a CUDA C++ example for converting an image to grayscale. This will demonstrate the fundamental concepts of CUDA programming, such as memory transfer between host and device, kernel execution, and parallel processing.

If you are interested in face detection, you would typically use a library like OpenCV, which has GPU-accelerated functions for face detection. OpenCV's cuda module provides GPU-accelerated implementations of many common image processing algorithms, including Haar cascade-based face detection.

Grayscale Conversion using CUDA C++
This example will take a color image (PPM format for simplicity, as it's easy to parse) and convert it to grayscale using CUDA.

# Prerequisites:
CUDA Toolkit: You need to have the NVIDIA CUDA Toolkit installed on your system.
NVIDIA GPU: A compatible NVIDIA GPU is required to run CUDA programs.
Basic C++ knowledge: Familiarity with C++ concepts is assumed.
Steps Involved:
Image Loading (Host): Load a color PPM image file into host memory.
Memory Allocation (Device): Allocate memory on the GPU for both the input color image and the output grayscale image.
Data Transfer (Host to Device): Copy the color image data from host memory to device memory.
Kernel Execution (Device): Launch a CUDA kernel to perform the grayscale conversion in parallel on the GPU.
Data Transfer (Device to Host): Copy the grayscale image data back from device memory to host memory.
Image Saving (Host): Save the grayscale image to a new PPM file.
Memory Deallocation: Free allocated memory on both host and device.
Grayscale Conversion Formula:
A common formula to convert RGB to grayscale is:
Gray=0.299×Red+0.587×Green+0.114×Blue

# Code
```
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm> // For std::min and std::max

// CUDA kernel to convert RGB to grayscale
__global__ void rgbToGrayscaleKernel(unsigned char* d_input_rgb, unsigned char* d_output_gray, int width, int height) {
    // Calculate the global thread index
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the current thread is within the image bounds
    if (col < width && row < height) {
        int index_rgb = (row * width + col) * 3; // 3 channels (R, G, B)
        int index_gray = row * width + col;      // 1 channel (Gray)

        // Get RGB values
        unsigned char r = d_input_rgb[index_rgb];
        unsigned char g = d_input_rgb[index_rgb + 1];
        unsigned char b = d_input_rgb[index_rgb + 2];

        // Apply grayscale conversion formula
        // Using integer arithmetic for efficiency and avoiding floating point issues
        // (0.299 * R + 0.587 * G + 0.114 * B) * 255.0 / 255.0
        // To avoid floating point, we can use integer scaled coefficients:
        // 77 * R + 150 * G + 29 * B (sum is 256)
        // This is equivalent to (0.3007 * R + 0.5859 * G + 0.1132 * B) * 256
        // Dividing by 256 gives a good approximation.
        unsigned char gray = static_cast<unsigned char>((77 * r + 150 * g + 29 * b) >> 8);

        d_output_gray[index_gray] = gray;
    }
}

// Function to load a PPM image (P6 format)
bool loadImagePPM(const std::string& filename, std::vector<unsigned char>& imageData, int& width, int& height, int& maxVal) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string magic;
    file >> magic;
    if (magic != "P6") {
        std::cerr << "Error: Invalid PPM magic number. Expected P6." << std::endl;
        return false;
    }

    file >> width >> height >> maxVal;
    if (width <= 0 || height <= 0 || maxVal <= 0 || maxVal > 255) {
        std::cerr << "Error: Invalid image dimensions or max color value." << std::endl;
        return false;
    }

    // Consume the newline character after reading maxVal
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    imageData.resize(width * height * 3); // 3 bytes per pixel (R, G, B)
    file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());

    if (!file) {
        std::cerr << "Error: Failed to read image data." << std::endl;
        return false;
    }

    file.close();
    return true;
}

// Function to save a grayscale PPM image (P5 format)
bool saveImagePPM(const std::string& filename, const std::vector<unsigned char>& imageData, int width, int height, int maxVal) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    file << "P5\n"; // P5 for grayscale
    file << width << " " << height << "\n";
    file << maxVal << "\n";

    file.write(reinterpret_cast<const char*>(imageData.data()), imageData.size());

    if (!file) {
        std::cerr << "Error: Failed to write image data." << std::endl;
        return false;
    }

    file.close();
    return true;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image.ppm> <output_image_gray.ppm>" << std::endl;
        return 1;
    }

    const std::string inputFilename = argv[1];
    const std::string outputFilename = argv[2];

    std::vector<unsigned char> h_input_rgb; // Host memory for input RGB image
    int width, height, maxVal;

    // Load the input image
    if (!loadImagePPM(inputFilename, h_input_rgb, width, height, maxVal)) {
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << ", MaxVal: " << maxVal << std::endl;

    size_t rgbImageSize = width * height * 3 * sizeof(unsigned char); // Size in bytes for RGB
    size_t grayImageSize = width * height * sizeof(unsigned char);     // Size in bytes for grayscale

    unsigned char* d_input_rgb;  // Device memory for input RGB image
    unsigned char* d_output_gray; // Device memory for output grayscale image

    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_input_rgb, rgbImageSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc for d_input_rgb failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMalloc(&d_output_gray, grayImageSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc for d_output_gray failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_rgb); // Clean up already allocated memory
        return 1;
    }

    // Copy input RGB image from host to device
    err = cudaMemcpy(d_input_rgb, h_input_rgb.data(), rgbImageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy HtoD failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_rgb);
        cudaFree(d_output_gray);
        return 1;
    }

    // Define grid and block dimensions
    // Typically, block dimensions are 16x16 or 32x32
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    rgbToGrayscaleKernel<<<gridSize, blockSize>>>(d_input_rgb, d_output_gray, width, height);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_rgb);
        cudaFree(d_output_gray);
        return 1;
    }

    // Synchronize to ensure kernel completes before copying back
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_rgb);
        cudaFree(d_output_gray);
        return 1;
    }

    // Copy grayscale image from device to host
    std::vector<unsigned char> h_output_gray(width * height); // Host memory for output grayscale image
    err = cudaMemcpy(h_output_gray.data(), d_output_gray, grayImageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy DtoH failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_rgb);
        cudaFree(d_output_gray);
        return 1;
    }

    // Save the grayscale image
    if (!saveImagePPM(outputFilename, h_output_gray, width, height, maxVal)) {
        cudaFree(d_input_rgb);
        cudaFree(d_output_gray);
        return 1;
    }

    std::cout << "Image successfully converted to grayscale and saved as " << outputFilename << std::endl;

    // Free device memory
    cudaFree(d_input_rgb);
    cudaFree(d_output_gray);

    return 0;
}
```
# Explanation of code
rgbToGrayscaleKernel (CUDA Kernel):

This is the core of the parallel processing. The __global__ specifier indicates that this function will be executed on the GPU by multiple threads.
d_input_rgb: Pointer to the input RGB image data on the device.
d_output_gray: Pointer to the output grayscale image data on the device.
width, height: Dimensions of the image.
Thread Indexing:
blockIdx.x, blockIdx.y: Identify the current block's coordinates within the grid.
blockDim.x, blockDim.y: Define the dimensions of each block.
threadIdx.x, threadIdx.y: Identify the current thread's coordinates within its block.
The global column (col) and row (row) are calculated to uniquely identify which pixel each thread should process.
Boundary Check: if (col < width && row < height) ensures that threads don't try to access memory outside the image boundaries, which can happen if the image dimensions are not perfect multiples of the block dimensions.
Grayscale Formula: Each thread calculates the grayscale value for its assigned pixel using the weighted average formula and stores it in the d_output_gray array. We use integer arithmetic (77 * r + 150 * g + 29 * b) >> 8 as a common optimization to avoid floating-point operations, where >> 8 is equivalent to dividing by 256.
loadImagePPM and saveImagePPM:

These are utility functions to handle reading and writing of PPM (Portable Pixmap) image files.
PPM P6 format: This is a simple binary format for color images. It starts with "P6", then width, height, maximum color value, and then the raw RGB bytes.
PPM P5 format: This is the grayscale binary format, similar to P6 but with "P5" and only one byte per pixel.
Error handling is included to check for file opening issues and invalid headers.
main Function (Host Code):

Argument Parsing: Takes input and output filenames as command-line arguments.
Host Memory (h_input_rgb, h_output_gray): std::vector<unsigned char> is used to store image data in the host (CPU) memory.
Device Memory (d_input_rgb, d_output_gray): cudaMalloc is used to allocate memory on the GPU (device). These pointers will hold the addresses of the allocated memory on the device.
cudaMemcpy: Used to transfer data between host and device.
cudaMemcpyHostToDevice: Copies data from host to device.
cudaMemcpyDeviceToHost: Copies data from device to host.
dim3 blockSize and dim3 gridSize: These define the dimensions of the thread blocks and the grid of blocks, respectively.
blockSize(16, 16): Each block will have 16×16=256 threads. This is a common and efficient block size.
gridSize: Calculated to cover the entire image. (width + blockSize.x - 1) / blockSize.x ensures that all pixels are processed, even if the image width isn't a perfect multiple of blockSize.x. This performs a ceiling division.
Kernel Launch: rgbToGrayscaleKernel<<<gridSize, blockSize>>>(...) launches the kernel. The <<<...>>> syntax is specific to CUDA and specifies the grid and block dimensions.
Error Checking: cudaGetLastError() is used after kernel launch to check for any asynchronous errors that might have occurred during kernel execution. cudaDeviceSynchronize() forces the CPU to wait until all GPU operations are complete, which is essential before copying results back to the host or freeing device memory.
Memory Management: cudaFree is used to deallocate device memory after it's no longer needed, preventing memory leaks on the GPU.
How to Compile and Run:
Save: Save the code above as grayscale_cuda.cu.

Create an Example PPM Image:
You'll need a .ppm image in P6 format. You can create a simple one manually (though it's tedious) or convert an existing image using tools like ImageMagick or GIMP.

Using ImageMagick (if installed): convert input.jpg rgb_image.ppm or convert input.png rgb_image.ppm
Manual (very simple 2x2 red image): Create a file named test_image.ppm with the following content:
```
P6
2 2
255
// Binary data starts here (red pixels)
// Red pixel 1
255 0 0
// Red pixel 2
255 0 0
// Red pixel 3
255 0 0
// Red pixel 4
255 0 0
```
(Note: For the manual creation, the actual binary data after "255" needs to be exactly 4 sets of 3 bytes for R, G, B values. You can't just type 255 0 0 literally, you need the actual byte values. It's much easier to use convert.)

# Compile: Use the NVIDIA CUDA compiler (nvcc):
```
nvcc grayscale_cuda.cu -o grayscale_cuda
```
# Run: Execute the compiled program, providing an input PPM and an output PPM filename:
```
./grayscale_cuda rgb_image.ppm grayscale_output.pp
```
Verify: Open grayscale_output.ppm using an image viewer (like GIMP, IrfanView, or your system's default image viewer) to confirm the conversion.

This example provides a solid foundation for understanding how to leverage CUDA for basic image processing tasks. For more complex operations like face detection, you would typically build upon existing libraries and frameworks designed for those purposes.
