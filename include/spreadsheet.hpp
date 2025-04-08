#ifndef SPREADSHEET_HPP
#define SPREADSHEET_HPP

#include <vector>
#include <string>

struct Cell {
	double value = 0.0;
	std::string forumla; // implement this later	
};

class SpreadsheetGrid {
private:
	std::vector<std::vector<Cell>> grid;
	size_t rows, cols;

public:

	SpreadsheetGrid(size_t r, size_t c);

	void setValue(size_t row, size_t col, double val);
	double getValue(size_t row, size_t col) const;
	void print() const;

	bool loadCSV(const std::string& filename);
	bool saveCSV(const std::string& filename) const; // doesnt change object state, just saves to file

	size_t rowCount() const { return rows; }
	size_t colCount() const { return cols; }

	// operations - cuda and CPU
	void applyScaleCUDA(double scale);
	void applyScaleCPU(double scale);
};

#endif
