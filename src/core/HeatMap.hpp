#ifndef HEATMAP_HPP
#define HEATMAP_HPP

#include "CartesianMatrix.hpp"
#include <QImage>
#include <QString>
#include <omp.h>

class HeatMap
{
public:
	static void createHeatMap(const CartesianMatrix<double>& matrix, const std::string& outputFilename)
	{
		int width  = matrix.getWidth();
		int height = matrix.getHeight();

		// Create a QImage with the same dimensions as the matrix
		QImage image(width, height, QImage::Format_RGB32);

		// Define the minimum and maximum values for the heatmap
		double minValue = 0.0;  // You can change this to be the minimum value of your matrix
		double maxValue = 1.0;  // You can change this to be the maximum value of your matrix

#pragma omp parallel for
		for(int i = 0; i < width; ++i) {
			for(int j = 0; j < height; ++j) {
				// Map the matrix value to a value between 0 and 255
				double value     = matrix.at({i, j});
				int    intensity = static_cast<int>(255.0 * ((value - minValue) / (maxValue - minValue)));
				intensity        = std::clamp(intensity, 0, 255);

				// Use the intensity value to set the color of the pixel
				QColor color;
				color.setHsv(intensity, 255, 255);
				image.setPixelColor(i, j, color);
			}
		}

		// Save the image to a file
		image.save(QString::fromStdString(outputFilename));
	}
};

#endif  // HEATMAP_HPP