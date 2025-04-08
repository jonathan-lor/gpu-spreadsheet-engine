#include "../include/spreadsheet.hpp"
#include <chrono>
#include <iostream>

using std::cout;

int main() {
	SpreadsheetGrid sheet(10000, 10000);

	sheet.setValue(0, 0, 3.58198);
	sheet.setValue(2, 2, 5.432);
	sheet.setValue(1, 2, 34.3);
	//sheet.print();

	cout << "timing cuda scale \n";
	
	auto start = std::chrono::high_resolution_clock::now();
	sheet.applyScaleCUDA(2);
	auto end = std::chrono::high_resolution_clock::now();
	// sheet.print();
	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	cout << "Execution time: " << duration.count() << " microseconds\n";

	cout << "timing cpu scale \n";

	auto start2 = std::chrono::high_resolution_clock::now();
	sheet.applyScaleCPU(2);
	auto end2 = std::chrono::high_resolution_clock::now();	
	// sheet.print();

	auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
	cout << "Execution time: " << duration2.count() << " microseconds\n";
	sheet.saveCSV("data/sheet.csv");

	SpreadsheetGrid loaded(1, 1);
	loaded.loadCSV("data/sheet.csv");
	
	cout << "Loaded sheet:\n";
	loaded.print();

	return 0;
	
}
