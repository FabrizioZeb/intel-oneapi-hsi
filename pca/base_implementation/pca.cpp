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
#define NUM_BANDS 128 //128 for Brain Images
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

    //Datos Paso 0
    FloatVector image_in_bp(NUM_BANDS*NUM_PIXELS);
    buffer buf_image_in_bp(image_in_bp);

    //Datos Paso 1
    FloatVector image_prima(NUM_BANDS * NUM_PIXELS);
    buffer buf_image_prima(image_prima);

    //CONTADOR DE TIEMPO
    auto start_k = std::chrono::high_resolution_clock::now();

    event e = q.submit([&](handler& h){

        //Accesor Entrada
        accessor acc_in(buf_image_in, h, read_only);
        
        //Accesor Resultado PCA
        accessor acc_out(buf_pca_out, h, read_write);

        //Accesor Paso 0
        accessor acc_image_in_bp(buf_image_in_bp, h, read_write);

        //Accesor Paso 1
        accessor acc_image_prima(buf_image_prima, h, read_write);

        h.single_task([=]()
            [[intel::scheduler_target_fmax_mhz(375)]] {

            //-------------------PASO 0 - imagen a Bands*Pixels----------------
            for (int i = 0; i < NUM_BANDS; i++) {
                for (int j = 0; j < NUM_PIXELS; j++) {
                    acc_image_in_bp[i * NUM_PIXELS + j] = acc_in[j * NUM_BANDS + i];
                }
            }

            //-----------------PASO 1 - Preprocesamiento ----------------------//
            float acc_mean_value[NUM_BANDS];
            for (int i = 0; i < NUM_BANDS; i++) {
                acc_mean_value[i] = 0;
                for (int j = 0; j < NUM_PIXELS; j++) {
                    acc_mean_value[i] += acc_image_in_bp[i * NUM_PIXELS + j];
                }
                acc_mean_value[i] = acc_mean_value[i] / NUM_PIXELS;
            }

            for (int i = 0; i < NUM_BANDS; i++) {
                for (int j = 0; j < NUM_PIXELS; j++) {
                    acc_image_prima[i * NUM_PIXELS + j] = acc_image_in_bp[i * NUM_PIXELS + j] - acc_mean_value[i];
                }
            }

            //----------------PASO 2 - Matriz Covarianzas---------------------//
            double acc_matrix_mult[NUM_BANDS * NUM_BANDS];
            for (int i = 0; i < NUM_BANDS; i++) {
                for (int j = i; j < NUM_BANDS; j++) {
                    for (int k = 0; k < NUM_PIXELS; k++) {
                        acc_matrix_mult[i * NUM_BANDS + j] += acc_image_prima[i * NUM_PIXELS + k] * acc_image_prima[j * NUM_PIXELS + k];
                    }
                    acc_matrix_mult[i * NUM_BANDS + j] = acc_matrix_mult[i * NUM_BANDS + j] / (NUM_PIXELS - 1);
                    acc_matrix_mult[j * NUM_BANDS + i] = acc_matrix_mult[i * NUM_BANDS + j];
                }
            }
            
            //----------------PASO 3 - Jacobi Method---------------------//
            int end_op = 0;
            int values_processed = 0;
            int iters = 0;
            double epsilon = 0.00001;
            
            int auxiliar_value_int;
            double sum_diag, stop, alpha, cosA, sinA, value_aux, value_aux_2, ordered;
            double a_ii, a_ij, a_jj, auxiliar_value;

            double acc_matrix_P[NUM_BANDS * NUM_BANDS];
            double acc_vector_eigenvals[NUM_BANDS];
            int acc_vector_order[NUM_BANDS];
            double acc_vector_eigenvecs_ordered[NUM_BANDS * NUM_PCA_BANDS];

            for (int i = 0; i < NUM_BANDS; i++) {
                for (int j = 0; j < NUM_BANDS; j++) {
                    if (i == j) {
                        acc_matrix_P[i * NUM_BANDS + j] = 1;
                    }
                    else {
                        acc_matrix_P[i * NUM_BANDS + j] = 0;
                    }
                }
            }

            while (end_op == 0) {
                end_op = 1;
                sum_diag = 0;
                for (int i = 0; i < NUM_BANDS; i++) {
                    sum_diag = sum_diag + acc_matrix_mult[i*NUM_BANDS+i];
                }
                for (int i = 0; i < NUM_BANDS; i++) {
                    for (int j = (i + 1); j < NUM_BANDS; j++) {
                        stop = epsilon * sum_diag;
                        if (acc_matrix_mult[i*NUM_BANDS+j] > stop || acc_matrix_mult[i*NUM_BANDS+j] < -stop) {
                            values_processed++;
                            end_op = 0;
                            a_ij = acc_matrix_mult[i*NUM_BANDS+j];
                            a_ii = acc_matrix_mult[i*NUM_BANDS+i];
                            a_jj = acc_matrix_mult[j*NUM_BANDS+j];

                            alpha = a_ij / (a_jj - a_ii);

                            cosA = 1 / (sqrt(alpha * alpha + 1));
                            sinA = cosA * alpha;

                            for (int k = 0; k < NUM_BANDS; k++) {
                                value_aux = cosA * acc_matrix_mult[i*NUM_BANDS+k] - sinA * acc_matrix_mult[j*NUM_BANDS+k];
                                value_aux_2 = sinA * acc_matrix_mult[i*NUM_BANDS+k] + cosA * acc_matrix_mult[j*NUM_BANDS+k];
                                acc_matrix_mult[i*NUM_BANDS+k] = value_aux;
                                acc_matrix_mult[j*NUM_BANDS+k] = value_aux_2;
                            }
                            for (int k = 0; k < NUM_BANDS; k++) {
                                value_aux = acc_matrix_mult[k*NUM_BANDS+i] * cosA - acc_matrix_mult[k*NUM_BANDS+j] * sinA;
                                value_aux_2 = acc_matrix_mult[k*NUM_BANDS+i] * sinA + acc_matrix_mult[k*NUM_BANDS+j] * cosA;
                                acc_matrix_mult[k*NUM_BANDS+i] = value_aux;
                                acc_matrix_mult[k*NUM_BANDS+j] = value_aux_2;

                                value_aux = acc_matrix_P[k*NUM_BANDS+i] * cosA - acc_matrix_P[k*NUM_BANDS+j] * sinA;
                                value_aux_2 = acc_matrix_P[k*NUM_BANDS+i] * sinA + acc_matrix_P[k*NUM_BANDS+j] * cosA;
                                acc_matrix_P[k*NUM_BANDS+i] = value_aux;
                                acc_matrix_P[k*NUM_BANDS+j] = value_aux_2;
                            }
                        }
                    }
                }
                iters++;
            }

            for (int i = 0; i < NUM_BANDS; i++) {
                acc_vector_eigenvals[i] = acc_matrix_mult[i*NUM_BANDS+i];
                acc_vector_order[i] = i;
            }

            ordered = 0;

            //Reordena los autovalores
            while (ordered == 0) {
                ordered = 1;
                for (int j = 1; j < NUM_BANDS; j++) {
                    if (acc_vector_eigenvals[j] > acc_vector_eigenvals[j - 1]) {
                        auxiliar_value = acc_vector_eigenvals[j];
                        acc_vector_eigenvals[j] = acc_vector_eigenvals[j - 1];
                        acc_vector_eigenvals[j - 1] = auxiliar_value;

                        auxiliar_value_int = acc_vector_order[j];
                        acc_vector_order[j] = acc_vector_order[j - 1];
                        acc_vector_order[j - 1] = auxiliar_value_int;
                        ordered = 0;
                    }
                }
            }

            for (int i = 0; i < NUM_PCA_BANDS; i++) {
                for (int j = 0; j < NUM_BANDS; j++) {
                    acc_vector_eigenvecs_ordered[j*NUM_PCA_BANDS+i] = acc_matrix_P[j * NUM_BANDS + acc_vector_order[i]];
                }
            }

            //------------------PASO 4 - Proyecciones -------------------//
            if (OPTION == 0) {
                for (int i = 0; i < NUM_PIXELS; i++) {
                    for (int j = 0; j < NUM_PCA_BANDS; j++) {
                        acc_out[i*NUM_PCA_BANDS+j] = 0;
                        for (int k = 0; k < NUM_BANDS; k++) {
                            acc_out[i*NUM_PCA_BANDS+j] = acc_out[i*NUM_PCA_BANDS+j] + (acc_image_prima[k*NUM_PIXELS+i] * (float)acc_vector_eigenvecs_ordered[k*NUM_PCA_BANDS+j]);
                        }
                    }
                }
            }
            else if (OPTION == 1) {
                for (int i = 0; i < NUM_PIXELS; i++) {
                    for (int j = 0; j < NUM_PCA_BANDS; j++) {
                        acc_out[i*NUM_PCA_BANDS+j] = 0;
                        for (int k = 0; k < NUM_BANDS; k++) {
                            acc_out[i*NUM_PCA_BANDS+j] = acc_out[i*NUM_PCA_BANDS+j] + (acc_image_in_bp[k*NUM_PIXELS+i] * acc_vector_eigenvecs_ordered[k*NUM_PCA_BANDS+j]);
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

    //pca_out = pcaOneBand(q, image_in_bp);
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
