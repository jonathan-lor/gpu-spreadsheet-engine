#include "../include/spreadsheet.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::getline;
using std::stringstream;
using std::stod;

SpreadsheetGrid::SpreadsheetGrid(size_t r, size_t c) : rows(r), cols(c), grid(r * c) {
	//grid.resize(rows, vector<Cell>(cols));
	// changing representation to flat array
}

SpreadsheetGrid::~SpreadsheetGrid() {
	// cout << "Calling destructor...\n";
}

void SpreadsheetGrid::setValue(size_t row, size_t col, double val) {
	if (row < rows && col < cols) {
		grid[row * cols + col].value = val;
	}
}

double SpreadsheetGrid::getValue(size_t row, size_t col) const {
	if (row < rows && col < cols) {
		return grid[row * cols + col].value;
	}
	return 0.0;
}

void SpreadsheetGrid::print() const {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			cout << grid[i * cols + j].value << "\t";	
		}
		cout << "\n";
	}
}

bool SpreadsheetGrid::loadCSV(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) return false;

    string line;
    grid.clear();
    rows = 0;
    cols = 0;

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        vector<Cell> rowCells;
        
        while (getline(ss, token, ',')) {
            Cell c;
            try {
                c.value = stod(token);
            } catch (...) {
                c.value = 0.0;
            }
            rowCells.push_back(c);
        }

        if (rows == 0) {
            cols = rowCells.size(); // Set expected column count
        } else if (rowCells.size() > cols) {
            return false; // Reject row with too many columns
        } else if (rowCells.size() < cols) {
            // Pad with zero-value cells
            rowCells.resize(cols, Cell{0.0});
        }

        grid.insert(grid.end(), rowCells.begin(), rowCells.end());
        rows++;
    }

    return true;
}


bool SpreadsheetGrid::saveCSV(const string& filename) const {
	ofstream file(filename);

	if (!file.is_open()) return false;

	for (size_t r = 0; r < rows; r++) {
		for (size_t c = 0; c < cols; c++) {
			file << grid[r * cols + c].value;
			if (c < cols - 1) file << ",";	
		}	
		file << "\n";
	}
	return true;
}
