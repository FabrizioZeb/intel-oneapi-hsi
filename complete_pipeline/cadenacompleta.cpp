//LIBRERIAS BASICAS
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <queue.hpp>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <time.h>

//LIBRERIAS FABRI
#include "parameters.h" //Contiene todas las constantes
//#include <fstream>
//#include <chrono>

//LIBRERIAS ENEKO
//#include "exception_handler.hpp"
//#include "functions.h"
/7#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

typedef std::vector<int> IntVector;
typedef std::vector<float> FloatVector;
typedef std::vector<double> DoubleVector;
typedef std::vector<char> CharVector;

class FeatureMatrixInitialize;
class FeatureMatrixDistance;
class SearchingStep;
class FilteringStep;
class VariableWindowKNNTOP;
class VariableWindowKNNBOT;
class FixedWindowKNN;

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
*  @param NUM_PIXELS	...
*  @param NUM_BANDS		...
*  @param normalizedImage Output
*/
void readNormalizedImage(char* image, FloatVector& normalizedImage) {
	FILE *fp;

	fp = fopen(image, "r");
	fread(normalizedImage.data(), sizeof(float), NUM_BANDS*NUM_PIXELS, fp);
	fclose(fp);
}

DoubleVector pcaOneBand(queue& q, const FloatVector &image_in){

    std::cout << "Entrada PCA: " << image_in[0] << "\n";

    //------------------PASO 0 - Inicializaci�n----------------------
    //Imagen de entrada
    buffer buf_image_in(image_in);

    //Resultado PCA
    DoubleVector pca_out(NUM_PIXELS*NUM_PCA_BANDS);
    buffer buf_pca_out(pca_out);

    //CONTADOR DE TIEMPO
    auto start_k = std::chrono::high_resolution_clock::now();

    event e = q.submit([&](sycl::handler& h){

        accessor acc_image_in(buf_image_in, h, read_only);
        
        //Accesor Resultado PCA
        accessor acc_out(buf_pca_out, h, write_only);

        h.single_task([=]()
            [[intel::scheduler_target_fmax_mhz(375)]] {
            
            //------------------PASO 0 - Inicialización----------------------
            double d[NUM_BANDS];
            double pixel_medio[NUM_BANDS];
            double CM[NUM_BANDS * NUM_BANDS];
            double VM[NUM_BANDS * NUM_BANDS]; //acc_matrix_mult

            for (int i = 0; i < NUM_BANDS; i++) {
                d[i] = 0;
                pixel_medio[i] = 0;
                for (int j = 0; j < NUM_BANDS; j++) {
                    CM[i*NUM_BANDS+j] = 0;
                    VM[i*NUM_BANDS+j] = 0;
                }
            }

            //-----------------PASO 1 - Cálculo de la matriz de correlación y pixel medio ----------------------//
            for (int i = 0; i < NUM_PIXELS; i++) {
                for (int k = 0; k < NUM_BANDS; k=k+8) {

                    double pix_med1 = pixel_medio[k];
                    double pix_med2 = pixel_medio[k+1];
                    double pix_med3 = pixel_medio[k+2];
                    double pix_med4 = pixel_medio[k+3];
                    double pix_med5 = pixel_medio[k+4];
                    double pix_med6 = pixel_medio[k+5];
                    double pix_med7 = pixel_medio[k+6];
                    double pix_med8 = pixel_medio[k+7];

                    double d_aux1 = acc_image_in[i * NUM_BANDS + k];
                    double d_aux2 = acc_image_in[i * NUM_BANDS + (k+1)];
                    double d_aux3 = acc_image_in[i * NUM_BANDS + (k+2)];
                    double d_aux4 = acc_image_in[i * NUM_BANDS + (k+3)];
                    double d_aux5 = acc_image_in[i * NUM_BANDS + (k+4)];
                    double d_aux6 = acc_image_in[i * NUM_BANDS + (k+5)];
                    double d_aux7 = acc_image_in[i * NUM_BANDS + (k+6)];
                    double d_aux8 = acc_image_in[i * NUM_BANDS + (k+7)];

                    d[k] = d_aux1;
                    d[k+1] = d_aux2;
                    d[k+2] = d_aux3;
                    d[k+3] = d_aux4;
                    d[k+4] = d_aux5;
                    d[k+5] = d_aux6;
                    d[k+6] = d_aux7;
                    d[k+7] = d_aux8;

                    pix_med1 += d_aux1;
                    pix_med2 += d_aux2;
                    pix_med3 += d_aux3;
                    pix_med4 += d_aux4;
                    pix_med5 += d_aux5;
                    pix_med6 += d_aux6;
                    pix_med7 += d_aux7;
                    pix_med8 += d_aux8;
                    
                    pixel_medio[k] = pix_med1;
                    pixel_medio[k+1] = pix_med2;
                    pixel_medio[k+2] = pix_med3;
                    pixel_medio[k+3] = pix_med4;
                    pixel_medio[k+4] = pix_med5;
                    pixel_medio[k+5] = pix_med6;
                    pixel_medio[k+6] = pix_med7;
                    pixel_medio[k+7] = pix_med8;
                }
                for (int j = 0; j < NUM_BANDS; j++) {
                    for (int k = 0; k < NUM_BANDS; k=k+8) {
                        CM[j*NUM_BANDS+k] += d[k] * d[j];
                        CM[j*NUM_BANDS+(k+1)] += d[k+1] * d[j];
                        CM[j*NUM_BANDS+(k+2)] += d[k+2] * d[j];
                        CM[j*NUM_BANDS+(k+3)] += d[k+3] * d[j];
                        CM[j*NUM_BANDS+(k+4)] += d[k+4] * d[j];
                        CM[j*NUM_BANDS+(k+5)] += d[k+5] * d[j];
                        CM[j*NUM_BANDS+(k+6)] += d[k+6] * d[j];
                        CM[j*NUM_BANDS+(k+7)] += d[k+7] * d[j];
                    }
                }
            }
            
            // Normalización de la matriz de correlación y cálculo final del pixel medio
            for (int i = 0; i < NUM_BANDS; i++) {
                pixel_medio[i] /= NUM_PIXELS;
                for (int j = 0; j < NUM_BANDS; j++) {
                    CM[i*NUM_BANDS+j] /= NUM_PIXELS;
                }
            }

            //----------------PASO 2 - Cálculo de la matriz de covarianza---------------------//
            for (int i = 0; i < NUM_BANDS; i++) {
                for (int j = 0; j < NUM_BANDS; j++) {
                    VM[i * NUM_BANDS + j] = CM[i * NUM_BANDS + j] - pixel_medio[i] * pixel_medio[j];
                }
            }
            
            //----------------PASO 3 - Power Iteration Method---------------------//
            double acc_dominant_eigenvector[NUM_BANDS];
            double epsilon = 1e-6;
            int max_iterations = 100;
        
            // Initialize the eigenvector with random values
            for (int i = 0; i < NUM_BANDS; i++) {
                acc_dominant_eigenvector[i] = 0.1 + 0.1 * i; // Simple initialization, can be improved
            }

            // Normalize the initial vector
            double norm = 0.0;
            for (int i = 0; i < NUM_BANDS; i++) {
                norm += acc_dominant_eigenvector[i] * acc_dominant_eigenvector[i];
            }
            norm = std::sqrt(norm);
            for (int i = 0; i < NUM_BANDS; i++) {
                acc_dominant_eigenvector[i] /= norm;
            }

            double eigenvalue = 0.0;
            double prev_eigenvalue = 0.0;
        
            for (int iter = 0; iter < max_iterations; iter++) {
                // Matrix-vector multiplication
                double new_vector[NUM_BANDS] = {0.0};
                for (int i = 0; i < NUM_BANDS; i++) {
                    for (int j = 0; j < NUM_BANDS; j++) {
                        new_vector[i] += VM[i * NUM_BANDS + j] * acc_dominant_eigenvector[j];
                    }
                }

                // Calculate new eigenvalue (Rayleigh quotient)
                eigenvalue = 0.0;
                for (int i = 0; i < NUM_BANDS; i++) {
                    eigenvalue += acc_dominant_eigenvector[i] * new_vector[i];
                }

                // Normalize the new vector
                norm = 0.0;
                for (int i = 0; i < NUM_BANDS; i++) {
                    norm += new_vector[i] * new_vector[i];
                }
                norm = std::sqrt(norm);
                for (int i = 0; i < NUM_BANDS; i++) {
                    acc_dominant_eigenvector[i] = new_vector[i] / norm;
                }

                // Check for convergence
                if (std::abs(eigenvalue - prev_eigenvalue) < epsilon) {
                    break;
                }
                prev_eigenvalue = eigenvalue;
            }

            // Store the dominant eigenvalue and eigenvector
            //acc_vector_eigenvals[0] = eigenvalue;
            //for (int i = 0; i < NUM_BANDS; i++) {
                //acc_vector_eigenvecs_ordered[i * NUM_PCA_BANDS] = acc_dominant_eigenvector[i];
            //}

            //------------------PASO 4 - Proyecciones -------------------//
            if (OPTION == 0) {
                for (int i = 0; i < NUM_PIXELS; i++) {
                    for (int j = 0; j < NUM_PCA_BANDS; j++) {
                        double acum = 0;
                        
                        #pragma unroll 4
                        for (int k = 0; k < NUM_BANDS; k++) {
                            acum += (acc_image_in[i * NUM_BANDS + k] - pixel_medio[k]) * acc_dominant_eigenvector[k];
                        }

                        acc_out[i*NUM_PCA_BANDS+j] = acum;
                    }
                }
            }
            else if (OPTION == 1) {
                for (int i = 0; i < NUM_PIXELS; i++) {
                    for (int j = 0; j < NUM_PCA_BANDS; j++) {
                        acc_out[i*NUM_PCA_BANDS+j] = 0;
                        for (int k = 0; k < NUM_BANDS; k++) {
                            //acc_out[i*NUM_PCA_BANDS+j] = acc_out[i*NUM_PCA_BANDS+j] + (acc_image_in_bp[k*NUM_PIXELS+i] * acc_vector_eigenvecs_ordered[k*NUM_PCA_BANDS+j]);
                        }
                    }
                }
            }
        });
    });
    q.wait();

    //CALCULO Y MUESTRO EL TIEMPO
	auto end_k = std::chrono::high_resolution_clock::now();
	auto duration_k = std::chrono::duration_cast<std::chrono::milliseconds>(end_k - start_k);
	std::cout << "PCA KERNEL Execution time: " << duration_k.count() << "milliseconds\n";

    return pca_out;
}

FloatVector svmPrediction(queue& q, const FloatVector &image_in) {
	printf("okayfun\n");

	//*******************************************************START STEP 0 **************************************************************//
	//*********************************************Variables declaration and initialization ************************************************//
	int  bufferSize = NUM_BANDS * sizeof(float);
	int i = 0;
	//char modelDir[255] = "";

	//Classification variables
	FILE* fp;
	int k, j;
	
	FloatVector w_vector_v(NUM_BINARY_CLASSIFIERS * NUM_BANDS);
	float** w_vector;
	w_vector = (float**)malloc(NUM_BANDS * sizeof(float*));
	for (int kk = 0; kk < NUM_BANDS; kk++) {
		w_vector[kk] = (float*)malloc(NUM_BINARY_CLASSIFIERS * sizeof(float));
	}

	float rho[NUM_BINARY_CLASSIFIERS];
	FloatVector rho_v(NUM_BINARY_CLASSIFIERS);
	FloatVector max_error_v(1);

	//Prob VARS
	float probA[NUM_BINARY_CLASSIFIERS];
	FloatVector probA_v(NUM_BINARY_CLASSIFIERS);
	float probB[NUM_BINARY_CLASSIFIERS];
	FloatVector probB_v(NUM_BINARY_CLASSIFIERS);

	float min_prob = 0.0000001;
	float max_prob = 0.9999999;

	//Other probs
	FloatVector pairwise_prob_v(NUM_CLASSES * NUM_CLASSES);
	FloatVector prob_estimates_v(NUM_CLASSES);
	FloatVector multi_prob_Q_v(NUM_CLASSES * NUM_CLASSES);
	FloatVector multi_prob_Qp_v(NUM_CLASSES);
	IntVector label_v(NUM_CLASSES);
	int* label;
	label = (int*)malloc(NUM_CLASSES * sizeof(int));
	FloatVector prob_estimates_result_ordered_v(NUM_PIXELS * NUM_CLASSES);
	CharVector predicted_labels_v(NUM_PIXELS);

	//-----------CARGA DE ARCHIVOS BINARIOS-----------------
	size_t reader;
	fp = fopen("../SVM_model/label.bin", "rb");
	for (j = 0; j < NUM_CLASSES; j++) {
		reader = fread(&label[j], sizeof(int), 1, fp);
		label_v[j] = label[j];
	}
	fclose(fp);

	fp = fopen("../SVM_model/ProbA.bin", "rb");
	for (j = 0; j < NUM_BINARY_CLASSIFIERS; j++) {
		reader = fread(&probA[j], sizeof(float), 1, fp);
		probA_v[j] = probA[j];
	}
	fclose(fp);

	fp = fopen("../SVM_model/ProbB.bin", "rb");
	for (j = 0; j < NUM_BINARY_CLASSIFIERS; j++) {
		reader = fread(&probB[j], sizeof(float), 1, fp);
		probB_v[j] = probB[j];
	}
	fclose(fp);

	fp = fopen("../SVM_model/rho.bin", "rb");
	for (j = 0; j < NUM_BINARY_CLASSIFIERS; j++) {
		reader = fread(&rho[j], sizeof(float), 1, fp);
		rho_v[j] = rho[j];
	}
	fclose(fp);

	fp = fopen("../SVM_model/w_vector.bin", "rb");
	for (j = 0; j < NUM_BINARY_CLASSIFIERS; j++) {
		for (k = 0; k < NUM_BANDS; k++) {
			reader = fread(&w_vector[k][j], sizeof(float), 1, fp);
			w_vector_v[NUM_BANDS * j + k] = w_vector[k][j];
		}
	}
	fclose(fp);

	//***************************************************************END STEP 0 ************************************************************//

	//**************************************************************START STEP 1 *********************************************************//
	//**************************************************************SVM Algorithm *********************************************//
	buffer buffer_w_vector(w_vector_v); //needed accessor
	buffer buffer_rho(rho_v); //needed accessor
	buffer buffer_probA(probA_v); //needed accessor
	buffer buffer_probB(probB_v); //needed accessor
	buffer buffer_label(label_v); //needed accessor
	buffer buffer_prob_estimates_result_ordered(prob_estimates_result_ordered_v); //needed accessor
	buffer buffer_image_in(image_in); //needed accessor

    //CONTADOR DE TIEMPO
    auto start_k = std::chrono::high_resolution_clock::now();

	event e = q.submit([&](sycl::handler& h) {

		accessor acc_w_vector = buffer_w_vector.get_access<cl::sycl::access::mode::read>(h);
		accessor acc_rho = buffer_rho.get_access<cl::sycl::access::mode::read>(h);
		accessor acc_probA = buffer_probA.get_access<cl::sycl::access::mode::read>(h);
		accessor acc_probB = buffer_probB.get_access<cl::sycl::access::mode::read>(h);
		accessor acc_label = buffer_label.get_access<cl::sycl::access::mode::read>(h);
		accessor acc_prob_estimates_result_ordered = buffer_prob_estimates_result_ordered.get_access<cl::sycl::access::mode::write>(h);
		accessor acc_image_in = buffer_image_in.get_access<cl::sycl::access::mode::read>(h);

		h.single_task([=]()
			[[intel::scheduler_target_fmax_mhz(375)]] {

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
				//float acc_prob_estimates_result[MAX_ESTIMATES_RESULT];
				float acc_epsilon = 0.005 / NUM_CLASSES;

				i = 0;

				for (int q = 0; q < NUM_PIXELS; q++) {
					p = 0;
					clasificador = 0;
					//fprintf(fg, "\n");
					//para todas las combinaciones de clases (clasificadores binarios) calculamos las probabilidades binarias
					for (b = 0; b < NUM_CLASSES; b++) {
						for (k = b + 1; k < NUM_CLASSES; k++) {
							sum1 = 0.0;
							for (j = 0; j < NUM_BANDS; j++) {
								sum1 += acc_image_in[q * NUM_BANDS + j] * acc_w_vector[clasificador * NUM_BANDS + j];
							}
							//acc_aux[0] = sum;
							//acc_aux2[0] = acc_w_vector[0*numberOfBands+1];

							acc_dec_values[clasificador] = sum1 - acc_rho[p];
							acc_sigmoid_prediction_fApB = acc_dec_values[clasificador] * acc_probA[p] + acc_probB[p];

							if (acc_sigmoid_prediction_fApB >= 0.0) {
								acc_sigmoid_prediction = exp(-acc_sigmoid_prediction_fApB) / (1.0 + exp(-acc_sigmoid_prediction_fApB));
							}
							else {
								acc_sigmoid_prediction = 1.0 / (1.0 + exp(acc_sigmoid_prediction_fApB));
							}
							//fprintf(fg, "%f\t", sigmoid_prediction);
							acc_pairwise_prob[b * NUM_CLASSES + k] = acc_sigmoid_prediction; //No se si va a funcionar bien con estos dos bucles anidados
							acc_pairwise_prob[k * NUM_CLASSES + b] = 1 - acc_sigmoid_prediction;

							p++;
							clasificador++;
						}
					}
					p = 0;

					//ORIGINAL FOR LOOP
					for (int b = 0; b < NUM_CLASSES; b++) {
						acc_prob_estimates[b] = 1.0 / NUM_CLASSES;
						float sum = 0.0f;
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

					while (stop == 0) {

						acc_pQp = 0.0;
						for (b = 0; b < NUM_CLASSES; b++) {
							acc_multi_prob_Qp[b] = 0.0;
							for (j = 0; j < NUM_CLASSES; j++) {
								acc_multi_prob_Qp[b] += acc_multi_prob_Q[b * NUM_CLASSES + j] * acc_prob_estimates[j];
							}
							acc_pQp += acc_prob_estimates[b] * acc_multi_prob_Qp[b];
						}
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
						if (stop == 0) {
							for (b = 0; b < NUM_CLASSES; b++) {
								acc_diff_pQp = (-acc_multi_prob_Qp[b] + acc_pQp) / (acc_multi_prob_Q[b * NUM_CLASSES + b]);
								acc_prob_estimates[b] = acc_prob_estimates[b] + acc_diff_pQp;
								acc_pQp = ((acc_pQp + acc_diff_pQp * (acc_diff_pQp * acc_multi_prob_Q[b * NUM_CLASSES + b] + 2 * acc_multi_prob_Qp[b])) / (1 + acc_diff_pQp)) / (1 + acc_diff_pQp);
								for (j = 0; j < NUM_CLASSES; j++) {
									acc_multi_prob_Qp[j] = (acc_multi_prob_Qp[j] + acc_diff_pQp * acc_multi_prob_Q[b * NUM_CLASSES + j]) / (1 + acc_diff_pQp);
									acc_prob_estimates[j] = acc_prob_estimates[j] / (1 + acc_diff_pQp);
								}
							}
						}
						iters++;

						if (iters == 100) {
							stop = 1;
						}
					}

					//elecci�n
					decision = 0;
					for (b = 1; b < NUM_CLASSES; b++) {
						if (acc_prob_estimates[b] > acc_prob_estimates[decision]) {
							decision = b;
						}
					}

					for (b = 0; b < NUM_CLASSES; b++) {
						position = acc_label[b] - 1;
						acc_prob_estimates_result_ordered[q * NUM_CLASSES + position] = acc_prob_estimates[b];
					}
				}

			});
		}); //END OF EVENT
	q.wait();

    //SVM PRINT TIME
	auto end_k = std::chrono::high_resolution_clock::now();
	auto duration_k = std::chrono::duration_cast<std::chrono::milliseconds>(end_k - start_k);
	std::cout << "SVM KERNEL Execution time: " << duration_k.count() << "milliseconds\n";

	return prob_estimates_result_ordered_v;
}

void KNNkernel(
	const DoubleVector& pcaOneBandResult, // PcaResult Input
	const FloatVector& svmProbabilityMapResult, // SVM Probability Map Result
	sycl::queue& q, //Device Queue
	CharVector& knn
) {

	const int safe_border_size_sample = SAFEBORDERSIZE * NUM_SAMPLES;
    
	//sycl::buffer<float, 1> Buff_input_pca_result(pcaOneBandResult);
	buffer Buff_input_pca_result(pcaOneBandResult);
	//sycl::buffer<float, 1> Buff_svmProbabilityMapResult(svmProbabilityMapResult); 
	buffer Buff_svmProbabilityMapResult(svmProbabilityMapResult);
	//sycl::buffer<char, 1> Buff_labelMap(knn);
	buffer Buff_labelMap(knn);

	//CONTADOR DE TIEMPO
    auto start_k = std::chrono::high_resolution_clock::now();

	q.submit([&](sycl::handler& h) {

		auto pca_accesor = Buff_input_pca_result.get_access<sycl::access::mode::read>(h);
		auto result_probs_SVM = Buff_svmProbabilityMapResult.get_access<sycl::access::mode::read>(h);
		auto labelMap = Buff_labelMap.get_access<sycl::access::mode::write>(h);
            
		h.single_task<FixedWindowKNN>([=]() [[intel::scheduler_target_fmax_mhz(480)]] {                        		            

        int zz;
        int winUEdge = 0;
        int winLEdge = safe_border_size_sample * 2;
        float PCA[NUM_PIXELS];// - (HALF_MAX_WS)];
	    float SVM_AUX[(NUM_PIXELS * 4)];// - HALF_MAX_WS) * 4];              
        //constexpr int REDUCED_SIZE = NUM_PIXELS - (2 * HALF_MAX_WS);

        for (int i = 0; i < NUM_PIXELS; i++){
			PCA[i] = pca_accesor[i];
            SVM_AUX[i*4]   = result_probs_SVM[i*4];
            SVM_AUX[i*4+1] = result_probs_SVM[i*4+1];
            SVM_AUX[i*4+2] = result_probs_SVM[i*4+2];
            SVM_AUX[i*4+3] = result_probs_SVM[i*4+3];
		}
        
        for (int i = HALF_MAX_WS; i < NUM_PIXELS - HALF_MAX_WS; i++) {
			int rIdx_i = i / NUM_SAMPLES;
			int cIdx_i = i % NUM_SAMPLES;
			float PCA_i = PCA[i];
			// Calculations for distance;
			featureDistance feat_dist[HALF_MAX_WS * 2]; // Es el tamaño máximo

			int pos_feat_dist = 0;
			
			float min;
			float last_min = 0;
			int neighbor[KNN]{};
			int neighbors[KNN];

			int jj = winUEdge;
			#pragma unroll 12
			for (int ii = 0; ii < MAX_WINDOWSIZE; ii++) {
			//for (int jj = winUEdge; jj < winLEdge; jj++) {
				int nextJJ = jj + 1;
				int rIdx_j = jj / NUM_SAMPLES;
				int cIdx_j = jj % NUM_SAMPLES;
				
				float dist1 = PCA_i - PCA[jj];

				int distr1 = (rIdx_i)-(rIdx_j);
				int distc1 = (cIdx_i)-(cIdx_j);

				float distance1 = (dist1 * dist1) + (distc1 * distc1) + (distr1 * distr1); 
				feat_dist[pos_feat_dist] = {distance1, jj};                    
				pos_feat_dist++;                    
				jj = nextJJ;
			}                
				
			for (int kk = 0; kk < KNN; kk++) {
				zz = 0;
				min = 1000000.0f;
				[[intel::initiation_interval(1)]]
				for (int ii = 0; ii < MAX_WINDOWSIZE; ii++) {
					if ((feat_dist[ii].distance > last_min) && (feat_dist[ii].distance <= min) && (feat_dist[ii].distance != 0)) {

						neighbor[zz] = feat_dist[ii].rc;                            
						if (min == feat_dist[ii].distance) {
							zz++;
						}
						else {
							zz = 0;
							min = feat_dist[ii].distance;
						}
					}                                                                      
				}
				last_min = min;
				[[intel::ivdep]]
				for (int x = 0; x <= zz; x++) {
					if ((kk + x) >= KNN) {
						break;
					}
					neighbors[kk + x] = neighbor[x];
				}
				kk += zz;
			}

			winLEdge++;
			winUEdge++;
        
        float filteredSVMap[NUM_CLASSES] = {0};

        float result_probs_SVM_map_c0 = 0.0f;
        float result_probs_SVM_map_c1 = 0.0f;
        float result_probs_SVM_map_c2 = 0.0f;
        float result_probs_SVM_map_c3 = 0.0f;
         
//POSIBLE PROBLEMA DE MEMORIA EN ESTE UNROLL
        #pragma unroll 2
        [[intel::speculated_iterations(10)]]
        for (int z = 0; z < KNN; z = z + 2){
            int kIdx1 = neighbors[z] + 1;
            int kIdx2 = neighbors[z + 1] + 1;

            float kidx1_filteredSVMap_c0 = 0.0f;
            float kidx1_filteredSVMap_c1 = 0.0f;
            float kidx1_filteredSVMap_c2 = 0.0f;
            float kidx1_filteredSVMap_c3 = 0.0f;

            float kidx2_filteredSVMap_c0 = 0.0f;
            float kidx2_filteredSVMap_c1 = 0.0f;
            float kidx2_filteredSVMap_c2 = 0.0f;
            float kidx2_filteredSVMap_c3 = 0.0f;

            if (kIdx1 < NUM_PIXELS) {
                int kIdxOffset1 = kIdx1 * NUM_CLASSES;

                kidx1_filteredSVMap_c0 = SVM_AUX[kIdxOffset1 + 0];
                kidx1_filteredSVMap_c1 = SVM_AUX[kIdxOffset1 + 1];
                kidx1_filteredSVMap_c2 = SVM_AUX[kIdxOffset1 + 2];
                kidx1_filteredSVMap_c3 = SVM_AUX[kIdxOffset1 + 3];

            }

            if (kIdx2 < NUM_PIXELS) {
                int kIdxOffset2 = kIdx2 * NUM_CLASSES;

                kidx2_filteredSVMap_c0 = SVM_AUX[kIdxOffset2 + 0];
                kidx2_filteredSVMap_c1 = SVM_AUX[kIdxOffset2 + 1];
                kidx2_filteredSVMap_c2 = SVM_AUX[kIdxOffset2 + 2];
                kidx2_filteredSVMap_c3 = SVM_AUX[kIdxOffset2 + 3];

            }                    
            result_probs_SVM_map_c0 += kidx1_filteredSVMap_c0 + kidx2_filteredSVMap_c0;
            result_probs_SVM_map_c1 += kidx1_filteredSVMap_c1 + kidx2_filteredSVMap_c1;
            result_probs_SVM_map_c2 += kidx1_filteredSVMap_c2 + kidx2_filteredSVMap_c2;
            result_probs_SVM_map_c3 += kidx1_filteredSVMap_c3 + kidx2_filteredSVMap_c3;
        }

        filteredSVMap[0] = result_probs_SVM_map_c0;
        filteredSVMap[1] = result_probs_SVM_map_c1;
        filteredSVMap[2] = result_probs_SVM_map_c2;
        filteredSVMap[3] = result_probs_SVM_map_c3;

        int maxProb = 0;
        [[intel::speculated_iterations(3)]]
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (filteredSVMap[c] > filteredSVMap[maxProb]) {
                maxProb = c;
            }
        }
        labelMap[i] = static_cast<char>(maxProb + 1);          
    }
    
	});
});
	q.wait();
	//KNN PRINT TIME
	auto end_k = std::chrono::high_resolution_clock::now();
	auto duration_k = std::chrono::duration_cast<std::chrono::milliseconds>(end_k - start_k);
	std::cout << "KNN KERNEL Execution time: " << duration_k.count() << "milliseconds\n";
};

int main(int argc, char* argv[]){

    #if FPGA_EMULATOR
	    auto d_selector = sycl::ext::intel::fpga_emulator_selector_v;
    #elif FPGA_SIMULATOR
	    auto d_selector = sycl::ext::intel::fpga_simulator_selector_v;
    #elif FPGA_HARDWARE
	    auto d_selector = sycl::ext::intel::fpga_selector_v;
    #else
	    auto d_selector = default_selector_v;
    #endif

    //--- QUEUE PREP ---
    // Print out the device information used for the kernel code.
    queue q(d_selector, exception_handler);
    std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";

    //--- IMAGE INPUT ---
    char* imagePath = "../Images/Op12C1.bin";
    FloatVector normalizedImage(NUM_PIXELS*NUM_BANDS);
    readNormalizedImage(imagePath, normalizedImage);

    //--- PCA STEP ---
	printf("Start PCA step...\n");
    DoubleVector pca_out(NUM_PIXELS*NUM_PCA_BANDS);
    pca_out = pcaOneBand(q, normalizedImage);

    std::cout << "Printing pca out...\n";
    FILE *fp_pca = fopen("resultado_PCA_FPGA.txt", "w");
    if(fp_pca != nullptr){
	    for(int i = 0; i<NUM_PIXELS;i++){
		    for(int j = 0; j<NUM_PCA_BANDS; j++){
			    fprintf(fp_pca, "%.6f\t", pca_out[i*NUM_PCA_BANDS+j]);
		    }
	    fprintf(fp_pca, "\n");
	    }
    }
    else{
	    std::cout<<"El archivo de output de pca no se ha podido abrir\n";
    }
	fclose(fp_pca);

    //--- SVM STEP ---
	printf("Start SVM step...\n");
    FloatVector svmProbabilityMapResult_v(NUM_PIXELS * NUM_CLASSES);
    svmProbabilityMapResult_v = svmPrediction(q, normalizedImage);

	// ---- PRINT SVM OUTPUT----
	printf("SVM done...\n");
	FILE* fp_svm = fopen("Output_SVM.txt", "w");
	for (int i = 0; i < NUM_PIXELS; i++) {
		for (int j = 0; j < NUM_CLASSES; j++) {
			fprintf(fp_svm, "%f\t", svmProbabilityMapResult_v[i * NUM_CLASSES + j]);
		}
		fprintf(fp_svm, "\n");
	}
	fclose(fp_svm);

	//--- KNN STEP ---
	printf("Start kNN step...\n");
	CharVector knnFilteredMap(NUM_PIXELS);

	KNNkernel(
		pca_out,
		svmProbabilityMapResult_v, 
		q,
		knnFilteredMap
	);    
    
	printf("KNN done...\n");

	FILE* fp_knn = fopen("Output_KNN_fgpaCompile.txt", "w");
	printf("knn Filtered Map -> -> %d y %d", (int)knnFilteredMap[0], (int)knnFilteredMap[37333])
	for (int i = 0; i < NUM_PIXELS; i++) {
		fprintf(fp_knn, "%d\t", (int)knnFilteredMap[i]);
		fprintf(fp_knn, "\n");
	}
	fclose(fp_knn);

    return 0;
}