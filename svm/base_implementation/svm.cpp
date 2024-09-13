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

//using namespace std;

/*
  Write the image "imagen" into a new image file "resultado_filename" with number of samples "num_samples", number of lines "num_lines"
  and number of bands "num_bands".
  This method does not write the header file .hdr, only the image file.
*/


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

	FILE *fp;
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
				fprintf(fp, "%.6f\t", normalizedImage[i*numberOfBands+k]);
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

//float** svmPrediction(FILE* f_time, int	numberOfLines, int	numberOfSamples, int numberOfBands, int	numberOfClasses, float* image_in) {

FloatVector svmPrediction(FILE* f_time, int	numberOfLines, int	numberOfSamples, int numberOfBands, int	numberOfClasses, FloatVector image_in) {
	printf("okayfun\n");
	float t_sec, t_usec;

	//t7s = clock();
	int  bufferSize = numberOfBands * sizeof(float);
	int numberOfPixels = numberOfLines * numberOfSamples;
	int i = 0;
	char modelDir[255] = "";
	//Classification variables
	FILE* fp;
	//FILE *f_out;
	//int j, k, p, q, b, clasificador, iters, stop, decision, position;
	int k, j;
	size_t reader;
	//float sum1;
	//float w_vector[100][NUM_BINARY_CLASSIFIERS];
	float** w_vector;
	FloatVector w_vector_v(NUM_BINARY_CLASSIFIERS * numberOfBands);
	float rho[NUM_BINARY_CLASSIFIERS];
	FloatVector rho_v(NUM_BINARY_CLASSIFIERS);
	//float dec_values[NUM_BINARY_CLASSIFIERS];
	FloatVector dec_values_v(NUM_BINARY_CLASSIFIERS, 0);
	//float max_error_aux;
	float sigmoid_prediction_fApB, sigmoid_prediction, pQp, max_error, diff_pQp;
	FloatVector sigmoid_prediction_fApB_v(1);
	FloatVector sigmoid_prediction_v(1);
	FloatVector pQp_v(1);
	FloatVector diff_pQp_v(1);
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
	float epsilon = 0.005 / NUM_CLASSES;
	FloatVector epsilon_v(1);
	epsilon_v[0] = epsilon;
	float** prob_estimates_result;
	char* predicted_labels;
	//LABEL VAR
	int* label;
	IntVector label_v(numberOfClasses);
	FloatVector aux(1);
	FloatVector aux2(1);

	float** prob_estimates_result_ordered;
	FloatVector prob_estimates_result_ordered_v(numberOfPixels * NUM_CLASSES);
	//f_out = fopen("output_SVM_SERIALE_label.txt", "w");

	w_vector = (float**)malloc(numberOfBands * sizeof(float*));
	for (int kk = 0; kk < numberOfBands; kk++) {
		w_vector[kk] = (float*)malloc(NUM_BINARY_CLASSIFIERS * sizeof(float));
	}

	//Initialize memory for the SVM probabilities matrix (pixels * classes)
	prob_estimates_result = (float**)malloc(numberOfPixels * sizeof(float*));
	FloatVector prob_estimates_result_v(numberOfPixels * numberOfClasses);
	//Initialize memory for the SVM probabilities ordered matrix (pixels * classes)
	prob_estimates_result_ordered = (float**)malloc(numberOfPixels * sizeof(float*));
	//FloatVector prob_estimates_result_ordered_v(numberOfPixels*NUM_CLASSES);
	//Initialize memory for the Samples Labels (classes)
	label = (int*)malloc(numberOfClasses * sizeof(int));
	//Load each pixel with all the bands in the memory
	for (i = 0; i < numberOfPixels; i++) {
		prob_estimates_result[i] = (float*)malloc(numberOfClasses * sizeof(float));
		prob_estimates_result_ordered[i] = (float*)malloc(numberOfClasses * sizeof(float));
	}
	predicted_labels = (char*)malloc(numberOfPixels * sizeof(char));
	CharVector predicted_labels_v(numberOfPixels);
	printf("okay9\n");

	//fp = fopen(strcat(modelDir, "label.bin"), "rb"); //I have commented this and added the following instruction because I have the .bin files in the Release folder
	//WITHOUT VECTOR LIBRARY
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
	std::cout << dec_values_v.size() << "\n";
	//QUEUES{
	//try{
	//queue qu(device_selector, dpc_common::exception_handler, property::queue::enable_profiling{});
	queue qu(d_selector, exception_handler);
	//CLASIFICACI�N
	//gettimeofday(&tv1,NULL);
	//para cada pixel
	//Preparar image_in

	buffer buffer_w_vector(w_vector_v);
	buffer buffer_dec_values(dec_values_v);
	buffer buffer_sigmoid_prediction_fApB(sigmoid_prediction_fApB_v);
	buffer buffer_sigmoid_prediction(sigmoid_prediction_v);
	buffer buffer_pairwise_prob(pairwise_prob_v);
	buffer buffer_prob_estimates(prob_estimates_v);
	buffer buffer_multi_prob_Q(multi_prob_Q_v);
	buffer buffer_multi_prob_Qp(multi_prob_Qp_v);
	buffer buffer_pQp(pQp_v);
	buffer buffer_diff_pQp(diff_pQp_v);
	buffer buffer_max_error(max_error_v);
	buffer buffer_prob_estimates_result(prob_estimates_result_v);
	buffer buffer_rho(rho_v);
	buffer buffer_probA(probA_v);
	buffer buffer_probB(probB_v);
	buffer buffer_label(label_v);
	buffer buffer_prob_estimates_result_ordered(prob_estimates_result_ordered_v);
	buffer buffer_image_in(image_in);
	buffer buffer_aux(aux);
	buffer buffer_aux2(aux2);
	buffer buffer_epsilon(epsilon_v);
	//buffer buffer_image_in(image_in);

	printf("okay7\n");

	event e = qu.submit([&](handler& h) {
		
		accessor acc_w_vector(buffer_w_vector, h, read_write);
		accessor acc_dec_values(buffer_dec_values, h, read_write);
		accessor acc_sigmoid_prediction_fApB(buffer_sigmoid_prediction_fApB, h, read_write);
		accessor acc_sigmoid_prediction(buffer_sigmoid_prediction, h, read_write);
		accessor acc_pairwise_prob(buffer_pairwise_prob, h, read_write);
		accessor acc_prob_estimates(buffer_prob_estimates, h, read_write);
		accessor acc_multi_prob_Q(buffer_multi_prob_Q, h, read_write);
		accessor acc_multi_prob_Qp(buffer_multi_prob_Qp, h, read_write);
		accessor acc_pQp(buffer_pQp, h, read_write);
		accessor acc_diff_pQp(buffer_diff_pQp, h, read_write);
		accessor acc_max_error(buffer_max_error, h, read_write);
		accessor acc_prob_estimates_result(buffer_prob_estimates_result, h, read_write);
		accessor acc_rho(buffer_rho, h, read_write);
		accessor acc_probA(buffer_probA, h, read_write);
		accessor acc_probB(buffer_probB, h, read_write);
		accessor acc_label(buffer_label, h, read_write);
		accessor acc_prob_estimates_result_ordered(buffer_prob_estimates_result_ordered, h, read_write);
		accessor acc_image_in(buffer_image_in, h, read_write);
		accessor acc_aux(buffer_aux, h, read_write);
		accessor acc_aux2(buffer_aux2, h, read_write);
		accessor acc_epsilon(buffer_epsilon, h, read_write);

		h.single_task([=]()
			[[intel::scheduler_target_fmax_mhz(375)]] {

//FloatVector acc_aux(1);

				int p, k, j, q, b, clasificador, iters, stop, decision, position, i;
				float sum1;
				float max_error_aux;

				i = 0;

				for (int q = 0; q < numberOfPixels; q++) {
					p = 0;
					clasificador = 0;
					//fprintf(fg, "\n");
					//para todas las combinaciones de clases (clasificadores binarios) calculamos las probabilidades binarias
					for (b = 0; b < NUM_CLASSES; b++) {
						for (k = b + 1; k < NUM_CLASSES; k++) {
							sum1 = 0.0;
							for (j = 0; j < numberOfBands; j++) {
								sum1 += acc_image_in[q * numberOfBands + j] * acc_w_vector[clasificador*numberOfBands+j];
							}
								//acc_aux[0] = sum;
								//acc_aux2[0] = acc_w_vector[0*numberOfBands+1];

							acc_dec_values[clasificador] = sum1 - acc_rho[p];
							acc_sigmoid_prediction_fApB[0] = acc_dec_values[clasificador] * acc_probA[p] + acc_probB[p];

							if (acc_sigmoid_prediction_fApB[0] >= 0.0) {
								acc_sigmoid_prediction[0] = exp(-acc_sigmoid_prediction_fApB[0]) / (1.0 + exp(-acc_sigmoid_prediction_fApB[0]));
							}
							else {
								acc_sigmoid_prediction[0] = 1.0 / (1.0 + exp(acc_sigmoid_prediction_fApB[0]));
							}
							if (acc_sigmoid_prediction[0] < min_prob) {
								acc_sigmoid_prediction[0] = min_prob;
							}
							if (acc_sigmoid_prediction[0] > max_prob) {
								acc_sigmoid_prediction[0] = max_prob;
							}
							//fprintf(fg, "%f\t", sigmoid_prediction);
							acc_pairwise_prob[b * NUM_CLASSES + k] = acc_sigmoid_prediction[0]; //No se si va a funcionar bien con estos dos bucles anidados
							acc_pairwise_prob[k * NUM_CLASSES + b] = 1 - acc_sigmoid_prediction[0];

							p++;
							clasificador++;
						}
					}
					p = 0;

					//ORIGINAL FOR LOOP
					for (b = 0; b < NUM_CLASSES; b++) {
						acc_prob_estimates[b] = 1.0 / NUM_CLASSES;
						acc_multi_prob_Q[b * NUM_CLASSES + b] = 0.0;
						for (j = 0; j < b; j++) {
							acc_multi_prob_Q[b * NUM_CLASSES + b] += acc_pairwise_prob[j * NUM_CLASSES + b] * acc_pairwise_prob[j * NUM_CLASSES + b];
							//multi_prob_Q[b][j] = multi_prob_Q[j][b];
							acc_multi_prob_Q[b * NUM_CLASSES + j] = acc_multi_prob_Q[j * NUM_CLASSES + b];
						}
						for (j = b + 1; j < NUM_CLASSES; j++) {
							/*multi_prob_Q[b][b] += pairwise_prob[j][b] * pairwise_prob[j][b];
							multi_prob_Q[b][j] = -pairwise_prob[j][b] * pairwise_prob[b][j];*/
							acc_multi_prob_Q[b * NUM_CLASSES + b] += acc_pairwise_prob[j * NUM_CLASSES + b] * acc_pairwise_prob[j * NUM_CLASSES + b];
							acc_multi_prob_Q[b * NUM_CLASSES + j] = -acc_pairwise_prob[j * NUM_CLASSES + b] * acc_pairwise_prob[b * NUM_CLASSES + j];
						}
					}


					/*stampa debug*/
					//for (b = 0; b < NUM_CLASSES; b++) {
					//	for (j = 0; j < NUM_CLASSES; j++) {
					//		fprintf(fg, "%f\t", multi_prob_Q[b][j]);
					//	}
					//}
					//fprintf(fg, "\n");

					iters = 0;
					stop = 0;

					while (stop == 0) {

						acc_pQp[0] = 0.0;
						for (b = 0; b < NUM_CLASSES; b++) {
							acc_multi_prob_Qp[b] = 0.0;
							for (j = 0; j < NUM_CLASSES; j++) {
								acc_multi_prob_Qp[b] += acc_multi_prob_Q[b * NUM_CLASSES + j] * acc_prob_estimates[j];
							}
							acc_pQp[0] += acc_prob_estimates[b] * acc_multi_prob_Qp[b];
						}
						acc_max_error[0] = 0.0;
						for (b = 0; b < NUM_CLASSES; b++) {
							max_error_aux = acc_multi_prob_Qp[b] - acc_pQp[0];
							if (max_error_aux < 0.0) {
								max_error_aux = -max_error_aux;
							}
							if (max_error_aux > acc_max_error[0]) {
								acc_max_error[0] = max_error_aux;
							}
						}
						if (acc_max_error[0] < acc_epsilon[0]) {
							stop = 1;
						}
						if (stop == 0) {
							for (b = 0; b < NUM_CLASSES; b++) {
								acc_diff_pQp[0] = (-acc_multi_prob_Qp[b] + acc_pQp[0]) / (acc_multi_prob_Q[b * NUM_CLASSES + b]);
								acc_prob_estimates[b] = acc_prob_estimates[b] + acc_diff_pQp[0];
								acc_pQp[0] = ((acc_pQp[0] + acc_diff_pQp[0] * (acc_diff_pQp[0] * acc_multi_prob_Q[b * NUM_CLASSES + b] + 2 * acc_multi_prob_Qp[b])) / (1 + acc_diff_pQp[0])) / (1 + acc_diff_pQp[0]);
								for (j = 0; j < NUM_CLASSES; j++) {
									acc_multi_prob_Qp[j] = (acc_multi_prob_Qp[j] + acc_diff_pQp[0] * acc_multi_prob_Q[b * NUM_CLASSES + j]) / (1 + acc_diff_pQp[0]);
									acc_prob_estimates[j] = acc_prob_estimates[j] / (1 + acc_diff_pQp[0]);
								}
							}
						}
						iters++;

						if (iters == 100) {
							stop = 1;
						}
					}

					for (b = 0; b < NUM_CLASSES; b++) {
						acc_prob_estimates_result[q * NUM_CLASSES + b] = acc_prob_estimates[b];
					}

					//elecci�n
					decision = 0;
					for (b = 1; b < NUM_CLASSES; b++) {
						if (acc_prob_estimates[b] > acc_prob_estimates[decision]) {
							decision = b;
						}
					}

					//predicted_labels[q] = (char)label[decision];  //ACTIVATE TO PRINT THE LABELS (otherwise we can sand label[decision] as result
					//fprintf(f_out, "%d\n", label[decision]);
				}



				/*stampa debug*/
				//for (b = 0; b < numberOfPixels; b++) {
				//	for (j = 0; j < NUM_CLASSES; j++) {
				//		fprintf(fg, "%f\t", prob_estimates_result[b][j]);
				//	}
				//	fprintf(fg, "\n");
				//}



				//stop_time = clock();
				//time_sim = (double)(stop_time - start_time) / CLOCKS_PER_SEC;
				//fprintf(tempi_simulazione, "SVM Algorithm - Serial code - Time simulation STEP 1---> %lf \n", time_sim);
				//*************************************************************************END STEP 1 **************************************************************************************

				//*************************************************************************START STEP 2 ************************************************************************************
				//************************************************************************Result ordered ***********************************************************************************
				//start_time = clock();

				//DOUBLE PARALLEL FOR IMPLEMENTATION

				

				//ORIGINAL DOUBLE FOR LOOP
				for (i = 0; i < numberOfPixels; i++) {
					for (j = 0; j < NUM_CLASSES; j++) {
						//position = label[j] - 1;
						position = acc_label[j] - 1; //VECTOR OK
						//prob_estimates_result_ordered[i][j] = prob_estimates_result[i][position];
						acc_prob_estimates_result_ordered[i * NUM_CLASSES + position] = acc_prob_estimates_result[i * NUM_CLASSES + j];
						//prob_estimates_result_ordered_v[i * NUM_CLASSES + position]=prob_estimates_result_v[i*NUM_CLASSES+j];
					}
				}
			});
		}); //END OF EVENT
	qu.wait();
	//t9e = clock();
	//fprintf(f_time, "%lf\n", (double)(t9e - t9s) / CLOCKS_PER_SEC);
	printf("okay8\n");
	std::cout << aux[0] << "\n";
	std::cout << aux2[0] << "\n";

	//FILE* fp;
	//int i, k;

	//PAIRWISE PROB PRINT
	/*fp = fopen("final_pairwise_prob.txt", "w");
	
	//std::ofstream fileW_matrix_mult("final_pairwise_prob.txt");
	for (int r = 0; r < NUM_CLASSES; r++) {
		for (int l = 0; l < NUM_CLASSES; l++) {
			//fileW_matrix_mult << std::setprecision(6) << std::fixed << pairwise_prob_v[r * NUM_CLASSES + l] << "\t";
			fprintf(fp, "%.6f\t", pairwise_prob_v[r * NUM_CLASSES + l]);
		}
		//fileW_matrix_mult << "\n";
		fprintf(fp, "\n");
	}
	//fileW_matrix_mult.close();
	fclose(fp);*/

	//MULTI PROB Q PRINT acc_multi_prob_Q
	/*fp = fopen("final_multi_prob_Q.txt", "w");

	//std::ofstream fileW_matrix_mult2("final_multi_prob_Q.txt");
	for (int r = 0; r < NUM_CLASSES; r++) {
		for (int l = 0; l < NUM_CLASSES; l++) {
			//fileW_matrix_mult2 << std::setprecision(6) << std::fixed << multi_prob_Q_v[r * NUM_CLASSES + l] << "\t";
			fprintf(fp, "%.6f\t", multi_prob_Q_v[r * NUM_CLASSES + l]);
		}
		//fileW_matrix_mult2 << "\n";
		fprintf(fp, "\n");
	}
	//fileW_matrix_mult2.close();
	fclose(fp);*/

	//IMAGE IN PRINT image_in
	/*std::ofstream fileW_matrix_mult3("final_image_in.txt");
	for (int r = 0; r < numberOfBands; r++) {
		for (int l = 0; l < numberOfPixels; l++) {
			fileW_matrix_mult3 << std::setprecision(6) << std::fixed << image_in[r * numberOfBands + l] << "\t";
		}
		fileW_matrix_mult3 << "\n";
	}
	fileW_matrix_mult3.close();*/

	//DEC VALUES dec_values
	/*fp = fopen("final_dec_values.txt", "w");
	//std::ofstream fileW_matrix_mult4("final_dec_values.txt");
	for (int r = 0; r < NUM_BINARY_CLASSIFIERS; r++) {
		
		//fileW_matrix_mult4 << std::setprecision(6) << std::fixed << dec_values_v[r] << "\t";
		fprintf(fp, "%.6f\t", dec_values_v[r]);

		//fileW_matrix_mult4 << "\n";
		fprintf(fp, "\n");
	}
	//fileW_matrix_mult4.close();
	fclose(fp);*/

	//RHO NUM_BINARY_CLASSIFIERS rho
	/*fp = fopen("final_rho.txt", "w");
	//std::ofstream fileW_matrix_mult5("final_rho.txt");
	for (int r = 0; r < NUM_BINARY_CLASSIFIERS; r++) {

		//fileW_matrix_mult5 << std::setprecision(6) << std::fixed << rho_v[r] << "\t";
		fprintf(fp, "%.6f\t", rho_v[r]);

		//fileW_matrix_mult5 << "\n";
		fprintf(fp, "\n");
	}
	//fileW_matrix_mult5.close();
	fclose(fp);*/

	////PROB ESTIMATES PRINT prob_estimates NUM_CLASSES
	/*fp = fopen("final_prob_estimates.txt", "w");
	//std::ofstream fileW_matrix_mult6("final_prob_estimates.txt");
	for (int r = 0; r < NUM_CLASSES; r++) {

		//fileW_matrix_mult6 << std::setprecision(6) << std::fixed << prob_estimates_v[r] << "\t";
		fprintf(fp, "%.6f\t", prob_estimates_v[r]);

		//fileW_matrix_mult6 << "\n";
		fprintf(fp, "\n");
	}
	//fileW_matrix_mult6.close();
	fclose(fp);*/

	////PROB ESTIMATES RESULT prob_estimates_result numberOfPixels NUM_CLASSES
	/*fp = fopen("final_prob_estimates_result.txt", "w");
	//std::ofstream fileW_matrix_mult7("final_prob_estimates_result.txt");
	for (int r = 0; r < numberOfPixels; r++) {
		for (int l = 0; l < numberOfClasses; l++) {
			//fileW_matrix_mult7 << std::setprecision(6) << std::fixed << prob_estimates_result_v[r * numberOfClasses + l] << "\t";
			fprintf(fp, "%.6f\t", prob_estimates_result_v[r * numberOfClasses + l]);
		}
		//fileW_matrix_mult7 << "\n";
		fprintf(fp, "\n");
	}
	//fileW_matrix_mult7.close();
	fclose(fp);*/

	//MULTI PROB Qp multi_prob_Qp
	/*fp = fopen("final_multi_prob_Qp.txt", "w");
	//std::ofstream fileW_matrix_mult8("final_multi_prob_Qp.txt");
	for (int r = 0; r < NUM_CLASSES; r++) {
		
		//fileW_matrix_mult8 << std::setprecision(6) << std::fixed << multi_prob_Qp_v[r] << "\t";
		fprintf(fp, "%.6f\t", multi_prob_Qp_v[r]);
		
		//fileW_matrix_mult8 << "\n";
		fprintf(fp, "\n");
	}
	//fileW_matrix_mult8.close();
	fclose(fp);*/

	//DIFF pQp diff_pQp
	/*std::ofstream fileW_matrix_mult8("final_diff_pQp.txt");
	for (int r = 0; r < NUM_CLASSES; r++) {

		fileW_matrix_mult8 << std::setprecision(6) << std::fixed << diff_pQp_v[r] << "\t";

		fileW_matrix_mult8 << "\n";
	}
	fileW_matrix_mult8.close();*/

	//acc_prob_estimates_result_ordered   numberOfPixels * NUM_CLASSES
	/*fp = fopen("final_prob_estimates_result_ordered.txt", "w");
	//std::ofstream fileW_matrix_mult9("final_prob_estimates_result_ordered.txt");
	for (int r = 0; r < numberOfPixels; r++) {
		for (int l = 0; l < numberOfClasses; l++) {
			//fileW_matrix_mult9 << std::setprecision(6) << std::fixed << prob_estimates_result_ordered_v[r * numberOfClasses + l] << "\t";
			fprintf(fp, "%.6f\t", prob_estimates_result_ordered_v[r * numberOfClasses + l]);
		}
		//fileW_matrix_mult9 << "\n";
		fprintf(fp, "\n");
	}
	//fileW_matrix_mult9.close();
	fclose(fp);*/

	//gettimeofday(&t9e, NULL);
	//t_sec = (float)(t9e.tv_sec - t9s.tv_sec);//
	//t_usec = (float)(t9e.tv_usec - t9s.tv_usec);//
	fprintf(f_time, "%lf\n", t_sec + t_usec / 1.0e+6);

	//stop_time = clock();
	//time_sim = (double)(stop_time - start_time) / CLOCKS_PER_SEC;
	//fprintf(tempi_simulazione, "SVM Algorithm - Serial code - Time simulation STEP 2---> %lf \n", time_sim);
	//*************************************************************************END STEP 2 **************************************************************************************

	//*************************************************************************START STEP 3 ************************************************************************************
	//*******************************************************************Free resources on CPU *********************************************************************************
	//start_time = clock();
	//t10s = clock();

	// FREE
	/*free(prob_estimates_result);
	free(label);
	free(predicted_labels);
	for (int kk = 0; kk < numberOfBands; kk++) {
		free(w_vector[kk]);
	}
	free(w_vector);*/

	//t10e = clock();
	//fprintf(f_time, "%lf\n", (double)(t10e - t10s) / CLOCKS_PER_SEC);
	//*************************************************************************END STEP 3 ************************************************************************************
	//stop_time = clock();
	//time_sim = (double)(stop_time - start_time) / CLOCKS_PER_SEC;
	//fprintf(tempi_simulazione, "SVM Algorithm - Serial code - Time simulation STEP 3---> %lf \n", time_sim);

	//fclose(f_out);
	//fclose(tempi_simulazione);
	//PRINT MATRIXES
	//PRINT MATRIX P

	//COMMMM
	/*std::ofstream fileW_matrix_mult("result_matrix_P_v3.txt");
	for (int r = 0; r < numberOfBands; r++) {
		for (int l = 0; j < numberOfBands; l++) {
			fileW_matrix_mult << std::setprecision(6) << std::fixed << matrix_P[r * NUM_BANDS + l] << "\t";
		}
		fileW_matrix_mult << "\n";
	}
	fileW_matrix_mult.close();*/


	return prob_estimates_result_ordered_v;
}

int main(int argc, char** argv) {
	printf("okay1\n");
	//std::cout << "okay" << std::endl;

	/*Time of the simulation*/
	//time_t t1s,t1e,t2s,t2e, t14s, t14e;
	//struct timeval t1s, t1e, t2s, t2e, t14s, t14e, t_pca1, t_pca2, t_svm1, t_svm2, t_knn1, t_knn2;
	//struct timeval start, end;
	float t_sec, t_usec;

	FILE* f_time;
	f_time = fopen("times.txt", "w");

	//printf("okay2");
	/* Image Data  */
	//uint16_t *rawImage;
	//float		*calibratedImage = NULL;

	//**Attention!! Please change the image name, the number of samples, lines and bands. ****//
	/*Data: Op20C1 S=329 L=377; Op15C1 S=493 L=375; Op25C2 S=402 L=472; Op12C1 S=496 L=442; Op8C1 S=548 L=459; Op8C2 S=552 L=479*/

	//t1s = clock();
	//gettimeofday(&t1s, NULL);

	//char *image = "../Images/P0C4_new.bin";
	char* image = "../x64/Images/Op12C1.bin";
	int	numberOfSamples = 496;
	int numberOfLines = 442;
	//int	numberOfBands = 826;
	//128
	printf("okay3\n");
	int	numberOfBands = 128; //after the pre-processing brain=128, derma=100   
	int numberOfPixels = (numberOfLines * numberOfSamples);
	float* normalizedImage = (float*)malloc(numberOfPixels * numberOfBands * sizeof(float));
	printf("okay4\n");
	/* PCA Data */
	double** pcaOneBandResult = NULL;
	int numberOfPcaBands = 1;
	/* SVM Data */
	//float** svmProbabilityMapResult = NULL;
	int numberOfClasses = 4;
	FloatVector svmProbabilityMapResult_v(numberOfPixels * NUM_CLASSES);
	/* KNN Data */
	char* knnFilteredMap = NULL;
	FloatVector normalizedImage_v(numberOfPixels * numberOfBands);
	// ---- READ RAW IMAGE ----
	//t_raw1 = clock();
	printf("okay5\n");

	//rawImage = readRawImageUint16(
	//	hsiCube2,
	//	numberOfLines,
	//	numberOfSamples,
	//	numberOfBands
	//);

	//t_raw2 = clock();


	// ---- IMAGE DARK AND WHITE CALIBRATION ----
	//t_pre1 = clock();

	//calibratedImage = rawImageCalibrationUint16(
	//	rawImage,		/* --> croppedRawImage*/
	//	numberOfLines,
	//	numberOfSamples,
	//	numberOfBands
	//);

	// ----IMAGE PREPROCESSING ----

	//normalizedImage = preProcessing(
	//	calibratedImage,
	//	numberOfLines,
	//	numberOfSamples,
	//	&numberOfBands
	//);



	//t_pre2 = clock();

	//printf("Time RAW IMAGE\t\t\t\t\t--->  %lf seconds \n", (double)(t_raw2 - t_raw1) / CLOCKS_PER_SEC);
	//printf("Time NORMALIZED IMAGE\t\t\t\t--->  %lf seconds \n", (double)(t_pre2 - t_pre1) / CLOCKS_PER_SEC);

	//t1e = clock();
	//gettimeofday(&t1e, NULL);
	//t_sec = (float)(t1e.tv_sec - t1s.tv_sec);
	//t_usec = (float)(t1e.tv_usec - t1s.tv_usec);

	//fprintf(f_time,"%lf\n", t_sec + t_usec/1.0e+6);

	//fflush(f_time);

	//t2s = clock();
	//gettimeofday(&t2s, NULL);
	
	// ---- Read the normalized image ----
	readNormalizedImage(
		image,
		numberOfPixels,
		numberOfBands,
		normalizedImage_v
	);
	//t2e = clock();
	//gettimeofday(&t2e, NULL);
	//t_sec = (float)(t2e.tv_sec - t2s.tv_sec);
	//t_usec = (float)(t2e.tv_usec - t2s.tv_usec);

	//fprintf(f_time,"%lf\n", t_sec + t_usec/1.0e+6);

	//fflush(f_time);
	printf("Upload Images...\n");


	// ---- WRITE FILE IMAGE ----

	//FILE	*fp;
	//fp = fopen("P0c1_normalizedImage.txt", "w");
	//for (int i = 0; i <numberOfPixels; i++) {
	//	for (int j = 0; j <numberOfBands; j++) {
	//		fprintf(fp, "%.6f\t", normalizedImage[(i*numberOfBands) + j]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);
	// ---- PCA ----

	//t_pca1 = clock();
	//gettimeofday(&t_pca1, NULL);

	//pcaOneBandResult = pcaOneBand(f_time, numberOfLines, numberOfSamples, numberOfBands, numberOfPcaBands, normalizedImage);

	//t_pca2 = clock();
	//gettimeofday(&t_pca2, NULL);
	//t_sec = (float)(t_pca2.tv_sec - t_pca1.tv_sec);
	//t_usec = (float)(t_pca2.tv_usec - t_pca1.tv_usec);
	//fprintf(f_time,"%lf\n", t_sec + t_usec/1.0e+6);
	//fflush(f_time);

	//printf("Time PCA \t\t\t\t--->  %lf seconds \n", (double)(t_pca2 - t_pca1) / CLOCKS_PER_SEC);
	//printf("PCA...\n");
	// ---- PRINT PCA OUTPUT----
	//printf("pca done\n");
	//FILE *fp_pca;
	//fp_pca = fopen("Output_PCA.txt", "w");
	//for (int i = 0; i < numberOfPixels; i++) {
	//	for (int j = 0; j < numberOfPcaBands; j++) {
	//		fprintf(fp_pca, "%lf\t", pcaOneBandResult[i][j]);
	//	}
	//	fprintf(fp_pca, "\n");
	//}
	//fclose(fp_pca);

	// ---- SVM ---- 
	//numberOfBands
	printf("okay6\n");
	auto start = std::chrono::high_resolution_clock::now();

	svmProbabilityMapResult_v = svmPrediction(f_time, numberOfLines, numberOfSamples, numberOfBands, numberOfClasses, normalizedImage_v);
	
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//t_sec = (float)(t_svm2.tv_sec - t_svm1.tv_sec);
	//t_usec = (float)(t_svm2.tv_usec - t_svm1.tv_usec);
	
	std::cout << "SVM Execution time: " << duration.count() << "milliseconds" << std::endl;

	// ---- PRINT SVM OUTPUT----
	printf("SVM done...\n");
	FILE* fp_svm;
	fp_svm = fopen("Output_SVM.txt", "w");
	for (int i = 0; i < numberOfPixels; i++) {
		for (int j = 0; j < numberOfClasses; j++) {
			//fprintf(fp_svm, "%f\t", svmProbabilityMapResult[i][j]);
			fprintf(fp_svm, "%f\t", svmProbabilityMapResult_v[i*numberOfClasses+j]);
		}
		fprintf(fp_svm, "\n");
	}
	fclose(fp_svm);



	/*Free memory resources*/
	//free(calibratedImage);
	free(normalizedImage);
	//free(pcaOneBandResult);
	//free(svmProbabilityMapResult);
	free(knnFilteredMap);

	fclose(f_time);
	//system("pause");
	return 1;

}