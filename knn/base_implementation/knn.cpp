// Fabrizio Nicolás Zeballos
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include "parameters.h"
#include <iomanip>
#include <fstream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <chrono>


#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif


using namespace sycl;
using namespace std;



class FeatureMatrixInitialize;
class FeatureMatrixDistance;
class SearchingStep;
class FilteringStep;


/**
 * @brief Performs the KNN step in the processing pipeline of the HSI Analysis
 * 
 * @param k_lines                       Height of matrix
 * @param k_samples_count               Width of matrix
 * @param k_bands_count                 Number of bands 
 * @param k_number_classes              Number of classes (SVM)
 * @param pcaOneBandResult              Input data from PCA results
 * @param svmProbabilityMapResult       Input data from SVM results
 * @param q                             queue
 * @param knn                           Output
 */
void KNNkernel(
	const int k_lines,
	const int k_samples_count,
	const int k_bands_count,
	const int k_number_classes,
	std::vector<double>& pcaOneBandResult, // PcaResult Input
	std::vector<float>& svmProbabilityMapResult, // SVM Probability Map Result
	sycl::queue& q, //Device Queue
	std::vector<char>& knn
) {

	int pixels = k_lines * k_samples_count;
	int pixels_local = pixels;
	int lines = k_lines;
	int samples = k_samples_count;
	int bands = k_bands_count;
	int classes = k_number_classes;
	int vector_length = pixels * bands;
	int safe_border_size_sample = SAFEBORDERSIZE * samples;

	std::vector<featureMatrixNode> feat_mx(pixels_local);
	std::vector<featureDistance> feat_dist(2 * safe_border_size_sample);
	std::vector<int> knnMatrix(pixels_local * KNN);
	std::vector<int> neighbor(KNN);
	std::vector<int> neighbors_host(KNN * pixels_local);


	std::vector<float> probClassMap(NUM_CLASSES * pixels_local);
	std::vector<char> labelMap(pixels_local);
	std::vector<float> _svmProbabilityMapResult = svmProbabilityMapResult;
	std::vector<float> filteredSVMap(pixels_local * NUM_CLASSES);

    // 	pixels_local = pixels; // Local calculation of KNN


	//Feature Matrix
	sycl::buffer<double, 1> input_pca_result(pcaOneBandResult);
	sycl::buffer<featureMatrixNode, 1> feat_mx_buffer(feat_mx);


	//Feature Distance
	sycl::buffer<featureDistance, 1> feat_dist_buffer(feat_dist);
	sycl::buffer<int, 1> knn_mx_buffer(knnMatrix);


	//=====================================//
   //=========     Filtering    ==========//
  //=====================================//

	sycl::buffer<float, 1> Buff_svmProbabilityMapResult(_svmProbabilityMapResult); //sycl::buffer<float, 1> Buff_svmProbabilityMapResult(svmProbabilityMapResult); 
	sycl::buffer<float, 1> Buff_filteredSVMap(filteredSVMap);
	sycl::buffer<float, 1> _probClasMap(probClassMap);
// 	sycl::buffer<char, 1> Buff_labelMap(labelMap);
    sycl::buffer<char, 1> Buff_labelMap(knn);


    printf("Starting...\n");
    auto start = std::chrono::high_resolution_clock::now();


	q.submit([&](sycl::handler& h) {

		auto pca_accesor = input_pca_result.get_access<sycl::access::mode::read>(h);
		auto feat_mx_accessor = feat_mx_buffer.get_access<sycl::access::mode::read_write>(h);

		//auto feat_mx_accessor = feat_mx_buffer.get_access<sycl::access::mode::read_write>(h);
		auto feat_dist_accessor = feat_dist_buffer.get_access<sycl::access::mode::read_write>(h);

		auto knn_mx_accessor = knn_mx_buffer.get_access<sycl::access::mode::read_write>(h);

		auto result_probs_SVM = Buff_svmProbabilityMapResult.get_access<sycl::access::mode::read>(h);
		auto filteredSVMap_accessor = Buff_filteredSVMap.get_access<sycl::access::mode::write>(h);
		auto labelMap = Buff_labelMap.get_access<sycl::access::mode::write>(h);
		auto probClasMap = _probClasMap.get_access<sycl::access::mode::write>(h);

		h.single_task<FeatureMatrixDistance>([=]() [[intel::kernel_args_restrict]] {

			int Idx = 0;
			for (int rIdx = 0; rIdx < lines; rIdx++) {
				for (int cIdx = 0; cIdx < samples; cIdx++) {
					feat_mx_accessor[Idx].PCA_pVal = pca_accesor[Idx];
					feat_mx_accessor[Idx].r = LAMBDA * rIdx;
					feat_mx_accessor[Idx].c = LAMBDA * cIdx;
					feat_mx_accessor[Idx].rc = rIdx * samples + cIdx;
					Idx++;
				}
			}
			
			int winUEdge = 0;
			int winLEdge = safe_border_size_sample;
			int zz;
			int knnMIdx = 0;

			for (int i = 0; i < pixels_local; i++) {
				// Calculations for distance;
				for (int j = winUEdge; j < winLEdge; j++) {

					double dist = feat_mx_accessor[i].PCA_pVal - feat_mx_accessor[j].PCA_pVal;
					int distr = feat_mx_accessor[i].r - feat_mx_accessor[j].r;
					int distc = feat_mx_accessor[i].c - feat_mx_accessor[j].c;

					double distance = (double)(dist * dist) + (double)(distc * distc) + (double)(distr * distr); // Euclidean distance
					feat_dist_accessor[j - winUEdge].distance = distance;
					feat_dist_accessor[j - winUEdge].rc = feat_mx_accessor[j].rc;
				}

				int windowSize = winLEdge - winUEdge;

				double min;
				double last_min = 0;
				int neighbor[KNN]{};
				int neighbors[KNN]{};

				for (int kk = 0; kk < KNN; kk++) {
					zz = 0;
					min = 1000000.0;
					for (int ii = 0; ii < windowSize; ii++) {
						if ((feat_dist_accessor[ii].distance > last_min) && (feat_dist_accessor[ii].distance <= min) && (feat_dist_accessor[ii].distance != 0)) {
							if (min == feat_dist_accessor[ii].distance) {
								zz++;
								neighbor[zz] = feat_dist_accessor[ii].rc;
							}
							else {
								zz = 0;
								min = feat_dist_accessor[ii].distance;
								neighbor[zz] = feat_dist_accessor[ii].rc;
							}
						}
					}
					last_min = min;
					for (int x = 0; x <= zz; x++) {
						if ((kk + x) >= KNN) {
							break;
						}
						neighbors[kk + x] = neighbor[x];

					}
					kk += zz;
				}

				if (i < safe_border_size_sample) {
					winLEdge++;
				}

				else if (i >= safe_border_size_sample && i < (pixels_local - safe_border_size_sample)) {
					winLEdge++;
					winUEdge++;
				}
				else {
					winUEdge++;
				}

				for (int auxI = 0; auxI < KNN; auxI++) {
					knn_mx_accessor[knnMIdx] = neighbors[auxI] + 1;
					knnMIdx++;
				}
			}

			int maxProb = 0;
			//int knnMIdx = 0;
			int kIdx = 0;
			// Transform SVM prob results to KNN input
			for (int aux = 0; aux < pixels_local; aux++) {
				for (int c = 0; c < NUM_CLASSES; c++) {
					probClasMap[c * pixels_local + aux] = result_probs_SVM[aux * NUM_CLASSES + c];
				}
			}

			knnMIdx = 0;
			//1.For each sample
			for (int i = 0; i < pixels_local; i++) {
				//2.For each class
				for (int c = 0; c < NUM_CLASSES; c++) {
					//3.For each of the KNN indexes of the current sample for the current class ...
					for (int z = 0; z < KNN; z++) {
						// ... accumulate
						kIdx = knn_mx_accessor[knnMIdx];
						knnMIdx++;
						if (kIdx < pixels_local)
							filteredSVMap_accessor[c * pixels_local + i] += probClasMap[c * pixels_local + kIdx];
					}
					// Compute average probability for the given pair {sample,class}
					filteredSVMap_accessor[c * pixels_local + i] /= (float)KNN;
					// Assign label corresponding to the highest prob. class
					// Rewind knnMIdx for a new iteration over the KNN of the current sample
					knnMIdx -= KNN;
				}
				maxProb = 0;
				for (int c = 1; c < NUM_CLASSES; c++) {
					if (filteredSVMap_accessor[c * pixels_local + i] > filteredSVMap_accessor[maxProb * pixels_local + i]) {
						maxProb = c;
					}
				}
				labelMap[i] = (char)maxProb + 1;
				knnMIdx += KNN;
			}



			});


		}).wait();
        
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "KNNkernel execution time: " << duration.count() << " milliseconds" << std::endl;

};


/**
 * @brief Read results from a text file that contains the data obtained from PCA step
 * 
 * @param result_filename 
 * @param numberOfSamples 
 * @param numberOfLines 
 * @param numberOfBands 
 * @return std::vector<double> 
 */
std::vector<double> readPcaResultVec(const char* result_filename, int numberOfSamples, int numberOfLines, int numberOfBands) {

	FILE* fp;

	int i, j;
	int np = numberOfLines * numberOfSamples;
	size_t vector_size = static_cast<size_t>(np) * numberOfBands;



	std::vector<double> pcaOneBandResult(vector_size);
	fp = fopen(result_filename, "r");
	if (fp != nullptr) {
		for (i = 0; i < np; i++) {
			for (j = 0; j < numberOfBands; j++) {
				fscanf(fp, "%lf", &pcaOneBandResult[i * numberOfBands + j]);
			}
		}
		fclose(fp);
	}
	else {
		cout << "Error Reading File: " << result_filename << std::endl;
	}


	//cout << "Length: " << pcaOneBandResult.size() << " & Size of file :" << sizeof(double) * pcaOneBandResult.size() / 1024.0 << "KB" << endl;

	return pcaOneBandResult;
}

/**
 * @brief Read results from a text file that contains the data obtained from SVM step
 * 
 * @param result_filename 
 * @param numberOfSamples 
 * @param numberOfLines 
 * @param numberOfBands 
 * @param numberOfClasses 
 * @return std::vector<float> 
 */
std::vector<float> readSVMResultVec(const char* result_filename, int numberOfSamples, int numberOfLines, int	numberOfBands, int	numberOfClasses) {

	int i, np = numberOfSamples * numberOfLines;
	std::vector<float> probabilityMap(np * numberOfClasses);

	FILE* fp_svm = fopen(result_filename, "r");
	if (fp_svm != nullptr) {
		for (i = 0; i < np; i++) {
			for (int j = 0; j < numberOfClasses; j++) {
				fscanf(fp_svm, "%f", &probabilityMap[i * numberOfClasses + j]);
			}
		}
		fclose(fp_svm);
	}
	else {
		std::cout << "Error Opening File: " << result_filename << std::endl;
	}

	return probabilityMap;
}


// Write the header into ".hdr" file
void writeHeader(const char* outHeader, int numberOfSamples, int numberOfLines, int numberOfBands) {
	// open the file
	FILE* fp = fopen(outHeader, "w+");
	fseek(fp, 0L, SEEK_SET);
	fprintf(fp, "ENVI\ndescription = {\nExported from MATLAB\n}\n");
	fprintf(fp, "samples = %d", numberOfSamples);
	fprintf(fp, "\nlines   = %d", numberOfLines);
	fprintf(fp, "\nbands   = %d", numberOfBands);
	fprintf(fp, "\ndata type = 5");
	fprintf(fp, "\ninterleave = bsq");
	fclose(fp);
}





int main() {



#if FPGA_EMULATOR
	auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
	auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
	auto selector = sycl::ext::intel::fpga_selector_v;
#else
	auto selector = default_selector_v;
#endif

	sycl::queue q(selector);
	auto device = q.get_device();
	std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;


	//clock_t startPCA, endPCA, startSVM, endSVM, startKNN, endKNN;
	int numberOfLines = 442;
	int numberOfSamples = 496;
	//int numberOfBands = 826;
	int numberOfBands = 128; //after the pre-processing brain = 128, derma = 100;
	int numberOfPixels = (numberOfLines * numberOfSamples);

	//PCA DATA
	std::vector<double> pcaOneBandResult(NUMBER_OF_PIXELS * NUMBER_OF_BANDS, 0.0);
	//double pcaOneBandResult[NUMBER_OF_PIXELS * NUMBER_OF_BANDS];
	int numberOfPcaBands = 1;

	//SVM DATA
	vector<float> svmProbabilityMapResult;
	int numberOfClasses = 4;

	//KNN DATA
	vector<char> knnFilteredMap(NUMBER_OF_PIXELS);

	std::fstream f_time;
	f_time.open("times.txt", std::fstream::out | std::fstream::trunc);


	printf("Start PCA read...\n");

	//startPCA = clock();

	pcaOneBandResult = readPcaResultVec("Output_PCA.txt", numberOfSamples, numberOfLines, numberOfPcaBands);

	//endPCA = clock();


	//std::cout << "Time PCA\t\t\t\t--->" << std::setprecision(5) << executionTime(startPCA,endPCA) << "seconds \n";


	// std::string pcaOutput = "./sections/PCA/Output_PCA_unidimensional.txt";
	// std::fstream fp_pca;
	// fp_pca.open(pcaOutput, std::fstream::out | std::fstream::trunc);

	// for (int i = 0; i < numberOfPixels; i++) {
// 		for (int j = 0; j < numberOfPcaBands; j++) {
			// fp_pca << std::setprecision(6) << std::fixed << pcaOneBandResult[i] << "\t";
			/*if(i < 1) cout << std::setprecision(6) << std::fixed << pcaOneBandResult[i * numberOfPcaBands + j] << "\n";*/
			//fprintf(fp_pca, "%lf\t", pcaOneBandResult[i][j]);
// 		}
		//fprintf(fp_pca, "\n");
		// fp_pca << "\n";
	// }

	// fp_pca.close();



	printf("Start SVM read...\n");

	//std::string svmOutput = "Output_SVM_fgpaCompile.txt";
	//std::string svmOutput = "Output_SVM.txt";
	std::string svmOutput = "./sections/SVM/Output_SVM_UnidimensialArray.txt";

	//startSVM = clock();
	svmProbabilityMapResult = readSVMResultVec("Output_SVM.txt", numberOfLines, numberOfSamples, numberOfBands, numberOfClasses);
	//endSVM = clock();

	//std::cout << "Time SVM\t\t\t\t--->" << std::setprecision(5) << executionTime(startSVM, endSVM) << "seconds \n";
	// ---- PRINT SVM OUTPUT----

	// std::fstream fp_svm;
	// fp_svm.open(svmOutput, std::fstream::out | std::fstream::trunc);
    // printf("Writing SVM...\n");
	// for (int i = 0; i < numberOfPixels; i++) {
	// 	for (int j = 0; j < numberOfClasses; j++) {
	// 		fp_svm << std::setprecision(6) << std::fixed << svmProbabilityMapResult[i*numberOfClasses + j] << "\t";
	// 	}
	// 	fp_svm << "\n";
	// }
	// fp_svm.close();

	// printf("Start kNN algo...\n");
	//startKNN = clock();
    
    
	KNNkernel(
		numberOfLines,
		numberOfSamples,
		numberOfBands,
		numberOfClasses,
		pcaOneBandResult,
		svmProbabilityMapResult,
		q,
		knnFilteredMap
	);
    
    
    //endKNN = clock();	
	//std::cout << "Time KNN\t\t\t\t--->" << std::setprecision(5) << executionTime(startKNN, endKNN) << "seconds \n";
    
    
    std::string knnOutput = "./sections/KNN/Output_KNN_fgpaCompile_float.txt";
		//std::string knnOutput = "Output_KNN.txt";

		FILE* fp_knn = fopen("./sections/KNN/Output_KNN_fgpaCompile_float.txt", "w");
		std::cout << "Escritura en fichero: " << knnOutput << "\n";

		if (fp_knn != nullptr) {
			for (int i = 0; i < NUMBER_OF_PIXELS; i++) {
				fprintf(fp_knn, "%d \n", (int)knnFilteredMap[i]);
			}
			fclose(fp_knn);
		}


	//endKNN = clock();	
	//std::cout << "Time KNN\t\t\t\t--->" << std::setprecision(5) << executionTime(startKNN, endKNN) << "seconds \n";

	//std::string knnOutput = "./sections/KNN/Output_KNN_fgpaCompile.txt";
	////std::string knnOutput = "Output_KNN.txt";

	//std::fstream fp_knn;
	//
	//fp_knn.open(knnOutput, std::fstream::out | std::fstream::trunc);
	//for (int i = 0; i < 2; i++) {
	//	fp_knn << (int)knnFilteredMap[i] << " \n";		
	//}
	//fp_knn.close();

	if (f_time.is_open())
		f_time.close();

	return 0;
}
