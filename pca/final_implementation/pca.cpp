// Antonio Álvarez Sánchez
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <queue.hpp>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <time.h>

using namespace sycl;

typedef std::vector<int> IntVector;
typedef std::vector<float> FloatVector;
typedef std::vector<double> DoubleVector;
typedef std::vector<char> CharVector;

#define OPTION 0
#define NUM_SAMPLES 496 //496 for PB1C1
#define NUM_LINES 442 //442 for PB1C1
#define NUM_BANDS 128 //128 for brain images
#define NUM_PCA_BANDS 1 //1
#define NUM_PIXELS 219232 //NUM_SAMPLES * NUM_LINES

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

//deberia ser doublevector
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

    event e = q.submit([&](handler& h){

        //Accesor Entrada PCA
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

    char* imagePath = "../Images/Op12C1.bin";

    FloatVector normalizedImage(NUM_PIXELS*NUM_BANDS);
    FloatVector image_in_bp(NUM_BANDS*NUM_PIXELS);
    DoubleVector pca_out(NUM_PIXELS*NUM_PCA_BANDS);

    readNormalizedImage(imagePath, normalizedImage);

    queue q(d_selector, exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";

    pca_out = pcaOneBand(q, normalizedImage);

    std::cout << "Printing pca out...\n";
    FILE *fp = fopen("resultado_PCA_FPGA.txt", "w");
    if(fp != nullptr){
	    for(int i = 0; i<NUM_PIXELS;i++){
		    for(int j = 0; j<NUM_PCA_BANDS; j++){
			    fprintf(fp, "%.6f\t", pca_out[i*NUM_PCA_BANDS+j]);
		    }
	    fprintf(fp, "\n");
	    }
    }
    else{
	    std::cout<<"El archivo de output de pca no se ha podido abrir\n";
    }

    return 0;
}
