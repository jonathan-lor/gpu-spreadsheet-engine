#include "../include/spreadsheet.hpp"
#include "../include/timing_helpers.hpp"
#include <iostream>
#include <memory>
#include <vector>

using std::vector;
using std::cout;
using std::make_unique;

int main() {
	auto sheet = make_unique<SpreadsheetGrid>(3, 3);

	sheet->setValue(0, 0, 3.58198);
	sheet->setValue(2, 2, 5.432);
	sheet->setValue(1, 2, 34.3);
	// sheet->print();

	// TIMING SCALE OPERATIONS
	cout << "Timing scale operations" << '\n';
	timer::timeFunction("CUDA scale", [&]() {
		sheet->applyScaleCUDA(2);
	});

	// sheet->print();

	timer::timeFunction("CPU scale", [&]() {
		sheet->applyScaleCPU(2);
	});

	// sheet->print();

	// TIMING ROW SUM OPERATIONS
	cout << "Timing Row Sum operations" << '\n';
	vector<double> rowSumsCUDA;
	timer::timeFunction("CUDA scale", [&]() {
		rowSumsCUDA = sheet->rowSumGPU();
	});

	// sheet->print();

	timer::timeFunction("CPU scale", [&]() {
		rowSumsCUDA = sheet->rowSumCPU();
	});

	// TIMING normalize OPERATIONS
	cout << "Timing Normalize operations" << '\n';
	timer::timeFunction("CUDA scale", [&]() {
		sheet->normalizeRowsGPU();
	});

	// sheet->print();

	timer::timeFunction("CPU scale", [&]() {
		sheet->normalizeRowsCPU();
	});

	sheet->print();


	sheet->saveCSV("data/sheet.csv");


	auto loaded = make_unique<SpreadsheetGrid>(0, 0);
	loaded->loadCSV("data/sheet.csv");

	cout << "Loaded sheet:\n";
	loaded->print();

	return 0;
}
