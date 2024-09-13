// Eneko Retolaza Ardanaz
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include "functions.h"
#include <queue.hpp>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "exception_handler.hpp"
#include <time.h>

using namespace sycl;

#include "dpc_common.hpp"

typedef std::vector<int> IntVector;
typedef std::vector<float> FloatVector;
typedef std::vector<double> DoubleVector;
typedef std::vector<char> CharVector;

void writeResult_matrix(double** image_out, const char* resultado_filename, int numberOfSamples, int numberOfLines, int numberOfBands) {
	FILE* fp;
	int i, j, np = numberOfSamples * numberOfLines;
	double* imagent = (double*)malloc(numberOfBands * np * sizeof(double));

	//open file "resultado_filename"
	if ((fp = fopen(resultado_filename, "wb")) != NULL)
	{
		fseek(fp, 0L, SEEK_SET);

		for (i = 0; i < np; i++) {
			for (j = 0; j < numberOfBands; j++) {
				imagent[i + j * np] = image_out[i][j];
			}
		}
		fwrite(imagent, 1, (np * numberOfBands * sizeof(double)), fp);
	}
	fclose(fp);
	free(imagent);
}

void writeResult_char(char* image_out, const char* resultado_filename, int numberOfSamples, int numberOfLines, int numberOfBands) {
	FILE* fp;
	int i, np = numberOfSamples * numberOfLines;
	double* imagent = (double*)malloc(np * sizeof(double));

	//open file "resultado_filename"
	if ((fp = fopen("knn", "wb")) != NULL)
	{
		fseek(fp, 0L, SEEK_SET);

		for (i = 0; i < np; i++) {
			imagent[i] = (double)image_out[i];
		}
		fwrite(imagent, 1, (np * sizeof(double)), fp);
	}
	fclose(fp);
	free(imagent);
}

// Write the header into ".hdr" file
void writeHeader(const char* outHeader, int numberOfSamples, int numberOfLines, int numberOfBands) {
	// open the file
	FILE* fp = fopen(outHeader, "w+");
	fseek(fp, 0L, SEEK_SET);
	fprintf(fp, "ENVI\ndescription = {\nExported from MATLAB}\n");
	fprintf(fp, "samples = %d", numberOfSamples);
	fprintf(fp, "\nlines   = %d", numberOfLines);
	fprintf(fp, "\nbands   = %d", numberOfBands);
	fprintf(fp, "\ndata type = 5");
	fprintf(fp, "\ninterleave = bsq");
	fclose(fp);
}

/* ************************************************************************   PRE PROCESSING *************************************************************** */
//Exception handler
// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
	for (std::exception_ptr const& e : e_list) {
		try {
			std::rethrow_exception(e);
		}
		catch (std::exception const& e) {
#if _DEBUG
			std::cout << "Failure" << std::endl;
#endif
			std::terminate();
		}
	}
};

/**
*  Read the pre-processed image:
*
*  @param image			Path where the image is stored
*  @param numberOfPixels	...
*  @param numberOfBands		...
*  @param normalizedImage Output
*/
void readNormalizedImage(char* image, int numberOfPixels, int numberOfBands, FloatVector& normalizedImage) {
	//std::ifstream file(image, std::ios::binary);
	//std::ofstream fileW("normalizedImage_v3.txt");

	FILE* fp;
	fp = fopen(image, "r");
	fread(normalizedImage.data(), sizeof(float), numberOfBands * numberOfPixels, fp);
	fclose(fp);

	fp = fopen("final_normalized_image", "w");
	if (fp != nullptr) {
		for (int i = 0; i < numberOfPixels; i++) {
			for (int k = 0; k < numberOfBands; k++) {
				//float value;
				//file.read(reinterpret_cast<char*>(&value), sizeof(float));
				//normalizedImage[i*numberOfBands + k] = value;
				//fileW << std::setprecision(6) << std::fixed << normalizedImage[i * NUM_BANDS + k] << "\t";
				fprintf(fp, "%.6f\t", normalizedImage[i * numberOfBands + k]);
			}
			fprintf(fp, "\n");
			//fileW << "\n";
		}
		fclose(fp);
	}
	//fileW.close();
}


/**
*  Perform the Averaging of a vector:
*
*  @param score			Input vector of values
*  @param initialValue	...
*  @param size			Size of the input vector
*/
float avg(float* score, int initialValue, int size)
{
	float total = 0;

	for (int i = initialValue; i < (initialValue + size); ++i) {
		total += score[i];
	}

	return (total / (float)size);
}

/**
*  Perform the Normalization of the Image:
*
*  @param vectorIn	Input Non-normalized vector
*  @param vectorOut	Output Normalized vector
*  @param yMin		Min normalization value
*  @param yMax		Max normalization value
*  @param size		Size of the input vector
*/
void mapMinMax(float* vectorIn, float* vectorOut, int yMin, int yMax, int size)
{
	float xMin = vectorIn[0];
	float xMax = vectorIn[0];

	for (int i = 1; i < size; ++i) {
		if (vectorIn[i] < xMin) {
			xMin = vectorIn[i];
		}
		if (vectorIn[i] > xMax) {
			xMax = vectorIn[i];
		}
	}
	for (int i = 0; i < size; ++i) {
		vectorOut[i] = (yMax - yMin) * (vectorIn[i] - xMin) / (xMax - xMin) + yMin;
	}

}

/* ************************************************************************  SVM  ********************************************************************* */
/**
*  Perform the SVM prediction over a entire hypercube:
*
*  @param numberOfLines	Number of pixels along X dimension
*  @param numberOfSamples	Number of pixels along Y dimension
*  @param numberOfBands 	Number of spectral bands
*  @param numberOfClasses 	Number of classes of the SVM model
*  @param image_in		 	HSI image matrix [px][bands]
*/

FloatVector svmPrediction(FILE* f_time, int	numberOfLines, int	numberOfSamples, int numberOfBands, int	numberOfClasses, FloatVector image_in) {
	printf("okayfun\n");

	//*************************************************************************START STEP 0 ************************************************************************************
	//*************************************************************Variables declaration and initialization ********************************************************************
	
	//Unnecessary variables are unused and declared within the kernel

	int  bufferSize = numberOfBands * sizeof(float);
	int numberOfPixels = numberOfLines * numberOfSamples;
	int i = 0;
	char modelDir[255] = "";
	//Classification variables
	FILE* fp;
	int k, j;
	size_t reader;
	float** w_vector;
	FloatVector w_vector_v(NUM_BINARY_CLASSIFIERS * numberOfBands);
	float rho[NUM_BINARY_CLASSIFIERS];
	FloatVector rho_v(NUM_BINARY_CLASSIFIERS);
	float sigmoid_prediction_fApB, sigmoid_prediction, pQp, max_error, diff_pQp;
	FloatVector max_error_v(1);

	//Prob VARS
	float probA[NUM_BINARY_CLASSIFIERS];
	FloatVector probA_v(NUM_BINARY_CLASSIFIERS);
	float probB[NUM_BINARY_CLASSIFIERS];
	FloatVector probB_v(NUM_BINARY_CLASSIFIERS);

	float min_prob = 0.0000001;
	float max_prob = 0.9999999;
	//Other probs
	float pairwise_prob[NUM_CLASSES][NUM_CLASSES];
	FloatVector pairwise_prob_v(NUM_CLASSES * NUM_CLASSES);
	float prob_estimates[NUM_CLASSES];
	FloatVector prob_estimates_v(NUM_CLASSES);
	float multi_prob_Q[NUM_CLASSES][NUM_CLASSES];
	FloatVector multi_prob_Q_v(NUM_CLASSES * NUM_CLASSES);
	float multi_prob_Qp[NUM_CLASSES];
	FloatVector multi_prob_Qp_v(NUM_CLASSES);
	float** prob_estimates_result;
	char* predicted_labels;
	//LABEL VAR
	int* label;
	IntVector label_v(numberOfClasses);

	float** prob_estimates_result_ordered;
	FloatVector prob_estimates_result_ordered_v(numberOfPixels * NUM_CLASSES);

	w_vector = (float**)malloc(numberOfBands * sizeof(float*));
	for (int kk = 0; kk < numberOfBands; kk++) {
		w_vector[kk] = (float*)malloc(NUM_BINARY_CLASSIFIERS * sizeof(float));
	}

	//Initialize memory for the SVM probabilities matrix (pixels * classes)
	prob_estimates_result = (float**)malloc(numberOfPixels * sizeof(float*));
	prob_estimates_result_ordered = (float**)malloc(numberOfPixels * sizeof(float*));
	label = (int*)malloc(numberOfClasses * sizeof(int));
	//Load each pixel with all the bands in the memory
	for (i = 0; i < numberOfPixels; i++) {
		prob_estimates_result[i] = (float*)malloc(numberOfClasses * sizeof(float));
		prob_estimates_result_ordered[i] = (float*)malloc(numberOfClasses * sizeof(float));
	}
	predicted_labels = (char*)malloc(numberOfPixels * sizeof(char));
	CharVector predicted_labels_v(numberOfPixels);
	printf("okay9\n");

	//auto d_selector{ sycl::ext::intel::fpga_emulator_selector_v };

	//fp = fopen(strcat(modelDir, "label.bin"), "rb"); //I have commented this and added the following instruction because I have the .bin files in the Release folder
	fp = fopen("../x64/SVM_model/label.bin", "rb");
	for (j = 0; j < NUM_CLASSES; j++) {
		reader = fread(&label[j], sizeof(int), 1, fp);
		label_v[j] = label[j];
	}
	fclose(fp);

	//fp = fopen(strcat(modelDir, "ProbA.bin"), "rb");//I have commented this and added the following instruction because I have the .bin files in the Release folder
	fp = fopen("../x64/SVM_model/ProbA.bin", "rb");
	for (j = 0; j < NUM_BINARY_CLASSIFIERS; j++) {
		reader = fread(&probA[j], sizeof(float), 1, fp);
		probA_v[j] = probA[j];
	}
	fclose(fp);

	//fp = fopen(strcat(modelDir, "ProbB.bin"), "rb");//I have commented this and added the following instruction because I have the .bin files in the Release folder
	fp = fopen("../x64/SVM_model/ProbB.bin", "rb");
	for (j = 0; j < NUM_BINARY_CLASSIFIERS; j++) {
		reader = fread(&probB[j], sizeof(float), 1, fp);
		probB_v[j] = probB[j];
	}
	fclose(fp);

	//fp = fopen(strcat(modelDir, "rho.bin"), "rb");//I have commented this and added the following instruction because I have the .bin files in the Release folder
	fp = fopen("../x64/SVM_model/rho.bin", "rb");
	for (j = 0; j < NUM_BINARY_CLASSIFIERS; j++) {
		reader = fread(&rho[j], sizeof(float), 1, fp);
		rho_v[j] = rho[j];
	}
	fclose(fp);

	//fp = fopen(strcat(modelDir, "w_vector.bin"), "rb");//I have commented this and added the following instruction because I have the .bin files in the Release folder
	fp = fopen("../x64/SVM_model/w_vector.bin", "rb");
	for (j = 0; j < NUM_BINARY_CLASSIFIERS; j++) {
		for (k = 0; k < numberOfBands; k++) {
			reader = fread(&w_vector[k][j], sizeof(float), 1, fp);
			w_vector_v[numberOfBands * j + k] = w_vector[k][j];
		}
	}
	fclose(fp);

	//*************************************************************************END STEP 0 **************************************************************************************

	//*************************************************************************START STEP 1 ************************************************************************************
	//************************************************************************SVM Algorithm ************************************************************************************

#if FPGA_EMULATOR
	auto d_selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
	auto d_selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
	auto d_selector = sycl::ext::intel::fpga_selector_v;
#else
	auto d_selector = default_selector_v;
#endif

	//auto d_selector{ sycl::ext::intel::fpga_emulator_selector_v };
	std::cout << image_in.size() << "\n";
	//QUEUES
	queue qu(d_selector, exception_handler);

	//BUFFERS
	buffer buffer_w_vector(w_vector_v); //needed accessor
	//buffer buffer_dec_values(dec_values_v); //not needed accessor
	//buffer buffer_sigmoid_prediction_fApB(sigmoid_prediction_fApB_v); //not needed accessor
	//buffer buffer_sigmoid_prediction(sigmoid_prediction_v); //not needed accessor
	//buffer buffer_pairwise_prob(pairwise_prob_v); //not needed accessor
	//buffer buffer_prob_estimates(prob_estimates_v); //not needed accessor
	//buffer buffer_multi_prob_Q(multi_prob_Q_v); //not needed accessor
	//buffer buffer_multi_prob_Qp(multi_prob_Qp_v); //not needed accessor
	//buffer buffer_pQp(pQp_v); //not needed accessor
	//buffer buffer_diff_pQp(diff_pQp_v); //not needed accessor
	//buffer buffer_max_error(max_error_v); //not needed accessor
	//buffer buffer_prob_estimates_result(prob_estimates_result_v); //declared within kernel
	buffer buffer_rho(rho_v); //needed accessor
	buffer buffer_probA(probA_v); //needed accessor
	buffer buffer_probB(probB_v); //needed accessor
	buffer buffer_label(label_v); //needed accessor
	buffer buffer_prob_estimates_result_ordered(prob_estimates_result_ordered_v); //needed accessor
	buffer buffer_image_in(image_in); //needed accessor
	//buffer buffer_epsilon(epsilon_v); //needed accessor
	//buffer buffer_image_in(image_in);

	printf("okay7\n");

	auto start_k = std::chrono::high_resolution_clock::now();

	event e = qu.submit([&](handler& h) {

		//ACCESSORS
		//accessor acc_w_vector(buffer_w_vector, h, read_write);
		accessor acc_w_vector = buffer_w_vector.get_access<cl::sycl::access::mode::read>(h);
		//accessor acc_dec_values(buffer_dec_values, h, read_write);
		//accessor acc_sigmoid_prediction_fApB(buffer_sigmoid_prediction_fApB, h, read_write);
		//accessor acc_sigmoid_prediction(buffer_sigmoid_prediction, h, read_write);
		//accessor acc_pairwise_prob(buffer_pairwise_prob, h, read_write);
		//accessor acc_prob_estimates(buffer_prob_estimates, h, read_write);
		//accessor acc_multi_prob_Q(buffer_multi_prob_Q, h, read_write);
		//accessor acc_multi_prob_Qp(buffer_multi_prob_Qp, h, read_write);
		//accessor acc_pQp(buffer_pQp, h, read_write);
		//accessor acc_diff_pQp(buffer_diff_pQp, h, read_write);
		//accessor acc_max_error(buffer_max_error, h, read_write);
		//accessor acc_prob_estimates_result(buffer_prob_estimates_result, h, read_write);
		//accessor acc_rho(buffer_rho, h, read_write);
		accessor acc_rho = buffer_rho.get_access<cl::sycl::access::mode::read>(h);
		//accessor acc_probA(buffer_probA, h, read_write);
		accessor acc_probA = buffer_probA.get_access<cl::sycl::access::mode::read>(h);
		//accessor acc_probB(buffer_probB, h, read_write);
		accessor acc_probB = buffer_probB.get_access<cl::sycl::access::mode::read>(h);
		//accessor acc_label(buffer_label, h, read_write);
		accessor acc_label = buffer_label.get_access<cl::sycl::access::mode::read>(h);
		//accessor acc_prob_estimates_result_ordered(buffer_prob_estimates_result_ordered, h, read_write); //only write
		accessor acc_prob_estimates_result_ordered = buffer_prob_estimates_result_ordered.get_access<cl::sycl::access::mode::write>(h);
		//accessor acc_image_in(buffer_image_in, h, read_write); //only read
		accessor acc_image_in = buffer_image_in.get_access<cl::sycl::access::mode::read>(h);
		//accessor acc_epsilon(buffer_epsilon, h, read_write); //only read
		//accessor acc_epsilon = buffer_epsilon.get_access<cl::sycl::access::mode::read>(h);

		h.single_task([=]()
			[[intel::scheduler_target_fmax_mhz(375)]] {

				// Variable declarations
				int p, k, j, q, b, clasificador, iters, stop, decision, position, i;
				float sum1;
				float max_error_aux;
				float acc_dec_values[NUM_BINARY_CLASSIFIERS];
				float acc_sigmoid_prediction_fApB, acc_sigmoid_prediction;
				float acc_pairwise_prob[NUM_CLASSES * NUM_CLASSES];
				float acc_prob_estimates[NUM_CLASSES];
				float acc_multi_prob_Q[NUM_CLASSES * NUM_CLASSES];
				float acc_multi_prob_Qp[NUM_CLASSES];
				float acc_pQp, acc_diff_pQp, acc_max_error;
				float acc_prob_estimates_result[MAX_ESTIMATES_RESULT];
				float acc_epsilon = 0.005 / NUM_CLASSES;

				i = 0;

				// Loop over each pixel
				for (int q = 0; q < numberOfPixels; q++) {
					p = 0;
					clasificador = 0;

					// Pairwise classifier loop for each class combination
					for (b = 0; b < NUM_CLASSES; b++) {
						for (k = b + 1; k < NUM_CLASSES; k++) {
							sum1 = 0.0;

							// Compute weighted sum for pixel bands
							for (j = 0; j < numberOfBands; j++) {
								sum1 += acc_image_in[q * numberOfBands + j] * acc_w_vector[clasificador * numberOfBands + j];
							}

							acc_dec_values[clasificador] = sum1 - acc_rho[p];

							// Sigmoid prediction calculation
							acc_sigmoid_prediction_fApB = acc_dec_values[clasificador] * acc_probA[p] + acc_probB[p];

							if (acc_sigmoid_prediction_fApB >= 0.0) {
								acc_sigmoid_prediction = exp(-acc_sigmoid_prediction_fApB) / (1.0 + exp(-acc_sigmoid_prediction_fApB));
							}
							else {
								acc_sigmoid_prediction = 1.0 / (1.0 + exp(acc_sigmoid_prediction_fApB));
							}

							// Assign probabilities for each class pair
							acc_pairwise_prob[b * NUM_CLASSES + k] = acc_sigmoid_prediction; //No se si va a funcionar bien con estos dos bucles anidados
							acc_pairwise_prob[k * NUM_CLASSES + b] = 1 - acc_sigmoid_prediction;

							p++;
							clasificador++;
						}
					}
					p = 0;

					// Initialize probability estimates for each class
					for (int b = 0; b < NUM_CLASSES; b++) {
						acc_prob_estimates[b] = 1.0 / NUM_CLASSES;
						float sum = 0.0f;

						// Compute multi-class probability matrix
						for (int j = 0; j < NUM_CLASSES; j++) {
							if (j < b) {
								sum += acc_pairwise_prob[j * NUM_CLASSES + b] * acc_pairwise_prob[j * NUM_CLASSES + b];
								acc_multi_prob_Q[b * NUM_CLASSES + j] = acc_multi_prob_Q[j * NUM_CLASSES + b];
							}
							else if (j > b) {
								sum += acc_pairwise_prob[j * NUM_CLASSES + b] * acc_pairwise_prob[j * NUM_CLASSES + b];
								acc_multi_prob_Q[b * NUM_CLASSES + j] = -acc_pairwise_prob[j * NUM_CLASSES + b] * acc_pairwise_prob[b * NUM_CLASSES + j];
							}
						}
						acc_multi_prob_Q[b * NUM_CLASSES + b] = sum;
					}

					iters = 0;
					stop = 0;

					// Iterative optimization loop
					while (stop == 0) {

						// Calculate Qp and pQp for optimization
						acc_pQp = 0.0;
						for (b = 0; b < NUM_CLASSES; b++) {
							acc_multi_prob_Qp[b] = 0.0;
							for (j = 0; j < NUM_CLASSES; j++) {
								acc_multi_prob_Qp[b] += acc_multi_prob_Q[b * NUM_CLASSES + j] * acc_prob_estimates[j];
							}
							acc_pQp += acc_prob_estimates[b] * acc_multi_prob_Qp[b];
						}

						// Check for convergence
						acc_max_error = 0.0;
						for (b = 0; b < NUM_CLASSES; b++) {
							max_error_aux = acc_multi_prob_Qp[b] - acc_pQp;
							if (max_error_aux < 0.0) {
								max_error_aux = -max_error_aux;
							}
							if (max_error_aux > acc_max_error) {
								acc_max_error = max_error_aux;
							}
						}
						if (acc_max_error < acc_epsilon) {
							stop = 1;
						}

						// If not converged, continue optimization
						if (stop == 0) {
							for (b = 0; b < NUM_CLASSES; b++) {
								acc_diff_pQp = (-acc_multi_prob_Qp[b] + acc_pQp) / (acc_multi_prob_Q[b * NUM_CLASSES + b]);
								acc_prob_estimates[b] = acc_prob_estimates[b] + acc_diff_pQp;
								acc_pQp = ((acc_pQp + acc_diff_pQp * (acc_diff_pQp * acc_multi_prob_Q[b * NUM_CLASSES + b] + 2 * acc_multi_prob_Qp[b])) / (1 + acc_diff_pQp)) / (1 + acc_diff_pQp);

								// Update probability estimates
								for (j = 0; j < NUM_CLASSES; j++) {
									acc_multi_prob_Qp[j] = (acc_multi_prob_Qp[j] + acc_diff_pQp * acc_multi_prob_Q[b * NUM_CLASSES + j]) / (1 + acc_diff_pQp);
									acc_prob_estimates[j] = acc_prob_estimates[j] / (1 + acc_diff_pQp);
								}
							}
						}
						iters++;

						// Limit iterations to 100
						if (iters == 100) {
							stop = 1;
						}
					}

					// Final decision based on max probability
					decision = 0;
					for (b = 1; b < NUM_CLASSES; b++) {
						if (acc_prob_estimates[b] > acc_prob_estimates[decision]) {
							decision = b;
						}
					}

					// Store final probability estimates ordered by class
					for (b = 0; b < NUM_CLASSES; b++) {
						position = acc_label[b] - 1;
						acc_prob_estimates_result_ordered[q * NUM_CLASSES + position] = acc_prob_estimates[b];
					}
				}

			});
		});
	qu.wait();

	auto end_k = std::chrono::high_resolution_clock::now();

	auto duration_k = std::chrono::duration_cast<std::chrono::milliseconds>(end_k - start_k);

	std::cout << "SVM KERNEL Execution time: " << duration_k.count() << "milliseconds" << std::endl;

	return prob_estimates_result_ordered_v;
}

int main(int argc, char** argv) {
	
	float t_sec, t_usec;

	FILE* f_time;
	f_time = fopen("times.txt", "w");

	char *image = "../Images/P0C4_new.bin";
	//char* image = "../x64/Images/Op12C1.bin";
	int	numberOfSamples = 496;
	int numberOfLines = 442;
	int	numberOfBands = 128; //after the pre-processing brain=128, derma=100   
	int numberOfPixels = (numberOfLines * numberOfSamples);
	float* normalizedImage = (float*)malloc(numberOfPixels * numberOfBands * sizeof(float));
	/* SVM Data */
	//float** svmProbabilityMapResult = NULL;
	int numberOfClasses = 4;
	FloatVector svmProbabilityMapResult_v(numberOfPixels * NUM_CLASSES);
	/* KNN Data */
	FloatVector normalizedImage_v(numberOfPixels * numberOfBands);
	// ---- READ RAW IMAGE ----
	//t_raw1 = clock();
	printf("okay5\n");


	//Read the normalized image
	readNormalizedImage(
		image,
		numberOfPixels,
		numberOfBands,
		normalizedImage_v
	);
	
	printf("Upload Images...\n");

	// ---- SVM ---- 
	//numberOfBands
	auto start = std::chrono::high_resolution_clock::now();

	svmProbabilityMapResult_v = svmPrediction(f_time, numberOfLines, numberOfSamples, numberOfBands, numberOfClasses, normalizedImage_v);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "SVM Execution time: " << duration.count() << "milliseconds" << std::endl;

	// ---- PRINT SVM OUTPUT----
	printf("SVM done...\n");
	FILE* fp_svm;
	fp_svm = fopen("Output_SVM.txt", "w");
	for (int i = 0; i < numberOfPixels; i++) {
		for (int j = 0; j < numberOfClasses; j++) {
			fprintf(fp_svm, "%f\t", svmProbabilityMapResult_v[i * numberOfClasses + j]);
		}
		fprintf(fp_svm, "\n");
	}
	fclose(fp_svm);



	/*Free memory resources*/
	free(normalizedImage);
	fclose(f_time);
	return 1;

}