#ifndef MATRIX
#define MATRIX

#include <vector>
#include <thread>
#include <algorithm>
#include <execution>
#include <iostream>
using namespace std;

template <typename T>
class Matrix {
private:
    int width;
    int height;
    std::pair<int, int> base{0, 0};
    std::vector<T> data;

public:
    struct Index {
        int x, y;
    };

    enum Direction {N, NE, E, SE, S, SW, W, NW}; 

public:
    Matrix(int width, int height, T initialValue = T()) : width(width), height(height) {
        data.resize(width * height, initialValue);
    }

    T& operator[](Index index) {
        int newX = (base.first + index.x + width) % width;
        int newY = (base.second + index.y + height) % height;
        return data[newY * width + newX];
    }

    void applyFunction(void(*func)(T&)) {
        std::for_each(std::execution::par, data.begin(), data.end(), func);
    }

    void shiftBase(Direction dir) {
        auto [dx, dy] = getShift(dir);
        base.first = (base.first + dx + width) % width;
        base.second = (base.second + dy + height) % height;
    }

public: // helper
    std::pair<int, int> getShift(Direction dir) {
        switch(dir) {
            case N:  return {0, -1};
            case NE: return {1, -1};
            case E:  return {1, 0};
            case SE: return {1, 1};
            case S:  return {0, 1};
            case SW: return {-1, 1};
            case W:  return {-1, 0};
            case NW: return {-1, -1};
            default: return {0, 0};
        }
    }

    void print() {
        std::cout << "---------------------- " << width << "x" << height << " ----------------------\n";
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                std::cout << (*this)[{j, i}];
                if (j != width - 1) std::cout << " | ";
            }
            std::cout << '\n';
        }
    }
};

#endif // Matrix