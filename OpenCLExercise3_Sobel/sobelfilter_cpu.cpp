#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

void print2DVector(std::vector<std::vector<int>> pixels){
        for (const auto& row : pixels) {
        for (const auto& pixel : row) {
            std::cout << pixel << " ";
        }
        std::cout << std::endl;
    }
}

void saveAsPGM(std::vector<std::vector<float>>& image, const std::string& filename) {
    int rows = image.size();
    int cols = image[0].size();
    const int maxVal = 255; // Maximum value for grayscale
    float maxGradient = 0;
    std::ofstream pgmFile(filename);
    if (!pgmFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            float magnitude = image[i][j];
            maxGradient = std::max(maxGradient, magnitude);
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = (image[i][j] * 255) / maxGradient;
            image[i][j] = val;
        }
    }
    // Write PGM header
    pgmFile << "P5\n";                     // PGM format
    pgmFile << cols << " " << rows << "\n"; // Width and Height
    pgmFile << maxVal << "\n";             // Maximum pixel value

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            char pixel = static_cast<char>(image[i][j]);
            pgmFile.write(&pixel, sizeof(pixel));
        }
    }
    // Write pixel values
    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         pgmFile << std::setw(3) << image[i][j] << " "; // Format output for readability
    //     }
    //     pgmFile << "\n"; // New line after each row
    // }

    pgmFile.close();
}
void sobelFilterCpu(std::vector<std::vector<float>> &input,
                    std::vector<std::vector<float>> &output,
                    std::vector<std::vector<int>> &hfilter,
                    std::vector<std::vector<int>> &vfilter, int width, int height){
    for(int i = 1;i<height-1;i++){
        for(int j = 1; j<width-1; j++){
            float tl = input[i-1][j-1];
            float ml = input[i][j-1];
            float bl = input[i+1][j+1];
       
            float tm = input[i-1][j];
            float mm = input[i][j];
            float bm = input[i+1][j];

            float tr = input[i-1][j+1];
            float mr = input[i][j+1];
            float br = input[i+1][j+1];

            float gx = 0;
            float gy = 0;

            gx = tl*hfilter[0][0] + ml*hfilter[1][0] + bl*hfilter[2][0] +
                 tm*hfilter[0][1] + mm*hfilter[1][1] + bm*hfilter[2][1] + 
                 tr*hfilter[0][2] + mr*hfilter[1][2] + br*hfilter[2][2];
        
            gy = tl*vfilter[0][0] + ml*vfilter[1][0] + bl*vfilter[2][0] +
                 tm*vfilter[0][1] + mm*vfilter[1][1] + bm*vfilter[2][1] + 
                 tr*vfilter[0][2] + mr*vfilter[1][2] + br*vfilter[2][2];
            output[i-1][j-1] = sqrt(gx*gx + gy*gy);
        }
    }
}

int main() {
    std::string filename = "Valve.pgm";
    std::ifstream file(filename, std::ios::binary);

    std::vector<std::vector<int>> hfilter = {{1, 0, -1},
                                            {2, 0, -2},
                                            {1, 0, -1}};

    std::vector<std::vector<int>> vfilter = {{1, 2, 1},
                                            {0, 0, 0},
                                            {-1, -2, -1}};

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return 1;
    }

    std::string format;
    int width, height, maxVal;

    // Read header
    file >> format;
    if (format != "P5") {
        std::cerr << "Error: Unsupported PGM format. Expected P5." << std::endl;
        return 1;
    }

    file >> width >> height >> maxVal;
    file.ignore(1);  // Ignore the newline character after maxVal

    // Check for 16-bit maxVal
    bool is16bit = maxVal > 255;

    // Allocate 2D array to store pixel values
    std::vector<std::vector<float>> pixels(height, std::vector<float>(width));
    std::vector<std::vector<float>> output(height, std::vector<float>(width));

    // Read pixel data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (is16bit) {
                // Read 2 bytes for 16-bit pixel value
                unsigned char byte1, byte2;
                file.read((char*)&byte1, 1);
                file.read((char*)&byte2, 1);
                pixels[i][j] = (byte1 << 8) | byte2;  // Combine bytes to form 16-bit value
            } else {
                // Read 1 byte for 8-bit pixel value
                unsigned char byte;
                file.read((char*)&byte, 1);
                pixels[i][j] = byte;
            }

            if (file.fail()) {
                std::cerr << "Error: Failed to read pixel data." << std::endl;
                return 1;
            }
        }
    }

    file.close();
    sobelFilterCpu(pixels, output, hfilter, vfilter, width, height);
    saveAsPGM(output, "outputcpu_float.pgm");
    // print2DVector(output);
    // std::cout << "Pixel Values:\n";
    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < 10; ++j) {
    //         std::cout << "Pixel[" << i << "][" << j << "] = " << output[i][j] << std::endl;
    //     }
    //     break;
    // }

    return 0;
}
