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

SpreadsheetGrid::SpreadsheetGrid(size_t r, size_t c) : rows(r), cols(c) {
	grid.resize(rows, vector<Cell>(cols));
}

void SpreadsheetGrid::setValue(size_t row, size_t col, double val) {
	if (row < rows && col < cols) {
		grid[row][col].value = val;
	}
}

double SpreadsheetGrid::getValue(size_t row, size_t col) const {
	if (row < rows && col < cols) {
		return grid[row][col].value;
	}
	return 0.0;
}

void SpreadsheetGrid::print() const {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			cout << grid[i][j].value << "\t";	
		}
		cout << "\n";
	}
}

bool SpreadsheetGrid::loadCSV(const string& filename) {
	ifstream file(filename);

	if(!file.is_open()) return false;

	string line;
	grid.clear();

	while (getline(file, line)) {
		stringstream ss(line);
		string token;
		vector<Cell> row;

		while(getline(ss, token, ',')) {
			Cell c;
			try {
				c.value = stod(token);
			} catch (...) {
				c.value = 0.0;
			}
			row.push_back(c);
		}

		grid.push_back(row);
	}

	rows = grid.size();
	cols = grid.empty() ? 0 : grid[0].size();
	return true;
}

bool SpreadsheetGrid::saveCSV(const string& filename) const {
	ofstream file(filename);

	if (!file.is_open()) return false;

	for (const auto& row : grid) {
		for (size_t i = 0; i < row.size(); i++) {
			file << row[i].value;
			if (i < row.size() - 1) file << ",";	
		}	
		file << "\n";
	}
	return true;
}
