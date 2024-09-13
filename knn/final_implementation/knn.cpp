// Fabrizio Nicolás Zeballos
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include "parameters.h"
#include <iomanip>
#include <fstream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif


using namespace sycl;
using namespace std;



class FeatureMatrixInitialize;
class FeatureMatrixDistance;
class SearchingStep;
class FilteringStep;
class VariableWindowKNNTOP;
class VariableWindowKNNBOT;
class FixedWindowKNN;


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
	const std::vector<float>& pcaOneBandResult, // PcaResult Input
	const std::vector<float>& svmProbabilityMapResult, // SVM Probability Map Result
	sycl::queue& q, //Device Queue
	std::vector<char>& knn
) {

	const int pixels = k_lines * k_samples_count;
	const int pixels_local = pixels;
	const int safe_border_size_sample = SAFEBORDERSIZE * SAMPLES;
    

	sycl::buffer<float, 1> Buff_input_pca_result(pcaOneBandResult);
	sycl::buffer<float, 1> Buff_svmProbabilityMapResult(svmProbabilityMapResult); 
	sycl::buffer<char, 1> Buff_labelMap(knn);
    
    // TOP WINDOW KNN
    q.submit([&](sycl::handler& h) {
        
        auto pca_accesor = Buff_input_pca_result.get_access<sycl::access::mode::read>(h);
		auto result_probs_SVM = Buff_svmProbabilityMapResult.get_access<sycl::access::mode::read>(h);
		auto labelMap = Buff_labelMap.get_access<sycl::access::mode::write>(h);
    
       h.single_task<VariableWindowKNNTOP>([=]() [[intel::scheduler_target_fmax_mhz(480)]] {
			int winLEdge_corners = safe_border_size_sample;
            int TOP_zz;
            float SVM_AUX_TOP[MAX_WINDOWSIZE * 4];
            
            for (int i = 0; i < HALF_MAX_WS; i++) {
                                                
                featureDistance TOP_feat_dist[MAX_WINDOWSIZE];
                          
                int TOP_pos_feat_dist = 0;
                int rIdx_i = i / SAMPLES;
                int cIdx_i = i % SAMPLES;
                
                float TOP_PCA_i = pca_accesor[i];
                
                int loadOffset = 0;
                for (int j = 0; j < winLEdge_corners; j++) {
                    
                    int nextTOP_Pos = TOP_pos_feat_dist + 1;
                    if(loadOffset < j) {
                        SVM_AUX_TOP[j*4] = result_probs_SVM[j*4];
                        SVM_AUX_TOP[j*4+1] = result_probs_SVM[j*4+1];
                        SVM_AUX_TOP[j*4+2] = result_probs_SVM[j*4+2];
                        SVM_AUX_TOP[j*4+3] = result_probs_SVM[j*4+3];
                        loadOffset++;
                    }
                                                
                    
                    //TOP
                    int rIdx_j = j;
                    int cIdx_j = j % SAMPLES;
                    rIdx_j /=  SAMPLES;
                    //TOP
                    float dist = TOP_PCA_i - pca_accesor[j];
                    int distr = (rIdx_i) - (rIdx_j); 
                    int distc = (cIdx_i) - (cIdx_j);                  
                    //TOP                                                 
                    float distance = (dist * dist) + (distc * distc) + (distr * distr);
                    TOP_feat_dist[TOP_pos_feat_dist].distance = distance;
                    TOP_feat_dist[TOP_pos_feat_dist].rc = j;
                    //NEXT ITER
                    TOP_pos_feat_dist = nextTOP_Pos;
                }
                
				float TOP_min;
				float TOP_last_min = 0;
				int TOP_neighbor[KNN]{};
				int TOP_neighbors[KNN]{};                
                
                for (int kk = 0; kk < KNN; kk++) {
					TOP_zz = 0;
					TOP_min = 1000000.0f;                    
					for (int ii = 0; ii < winLEdge_corners; ii++) {
						if ((TOP_feat_dist[ii].distance > TOP_last_min) && (TOP_feat_dist[ii].distance <= TOP_min) && (TOP_feat_dist[ii].distance != 0)) {
							TOP_neighbor[TOP_zz] = TOP_feat_dist[ii].rc;                            
                            if (TOP_min == TOP_feat_dist[ii].distance) {
								TOP_zz++;
							}
							else {
								TOP_zz = 0;
								TOP_min = TOP_feat_dist[ii].distance;
							}
						}                                                                      
					}
					TOP_last_min = TOP_min;
					for (int x = 0; x <= TOP_zz; x++) {
						if ((kk + x) >= KNN) {
							break;
						}
						TOP_neighbors[kk + x] = TOP_neighbor[x];
					}
					kk += TOP_zz;
				}
                                
                winLEdge_corners++;				
                
                float TOP_filteredSVMap[NUM_CLASSES] = {0};              
                
                float TOP_result_probs_SVM_map_c0 = 0.0f;
                float TOP_result_probs_SVM_map_c1 = 0.0f;
                float TOP_result_probs_SVM_map_c2 = 0.0f;
                float TOP_result_probs_SVM_map_c3 = 0.0f;                
                
                [[intel::speculated_iterations(20)]]
                for (int z = 0; z < KNN; z = z + 2){
                    int kIdx1 = TOP_neighbors[z] + 1;
                    int kIdx2 = TOP_neighbors[z + 1] + 1;
                    
                    float kidx1_filteredSVMap_c0 = 0.0f;
                    float kidx1_filteredSVMap_c1 = 0.0f;
                    float kidx1_filteredSVMap_c2 = 0.0f;
                    float kidx1_filteredSVMap_c3 = 0.0f;
                    
                    float kidx2_filteredSVMap_c0 = 0.0f;
                    float kidx2_filteredSVMap_c1 = 0.0f;
                    float kidx2_filteredSVMap_c2 = 0.0f;
                    float kidx2_filteredSVMap_c3 = 0.0f;
                    
                    
                    if (kIdx1 < pixels_local) {
                        int kIdxOffset1 = kIdx1 * NUM_CLASSES;                        
                        kidx1_filteredSVMap_c0 = SVM_AUX_TOP[kIdxOffset1 + 0];
                        kidx1_filteredSVMap_c1 = SVM_AUX_TOP[kIdxOffset1 + 1];
                        kidx1_filteredSVMap_c2 = SVM_AUX_TOP[kIdxOffset1 + 2];
                        kidx1_filteredSVMap_c3 = SVM_AUX_TOP[kIdxOffset1 + 3];                        
                    }
                    
                    if (kIdx2 < pixels_local) {
                        int kIdxOffset2 = kIdx2 * NUM_CLASSES;
                        kidx2_filteredSVMap_c0 = SVM_AUX_TOP[kIdxOffset2 + 0];
                        kidx2_filteredSVMap_c1 = SVM_AUX_TOP[kIdxOffset2 + 1];
                        kidx2_filteredSVMap_c2 = SVM_AUX_TOP[kIdxOffset2 + 2];
                        kidx2_filteredSVMap_c3 = SVM_AUX_TOP[kIdxOffset2 + 3];                        
                    }                    
 

                    TOP_result_probs_SVM_map_c0 += kidx1_filteredSVMap_c0 + kidx2_filteredSVMap_c0;                    
                    TOP_result_probs_SVM_map_c1 += kidx1_filteredSVMap_c1 + kidx2_filteredSVMap_c1;                    
                    TOP_result_probs_SVM_map_c2 += kidx1_filteredSVMap_c2 + kidx2_filteredSVMap_c2;                    
                    TOP_result_probs_SVM_map_c3 += kidx1_filteredSVMap_c3 + kidx2_filteredSVMap_c3;
                }
                
                TOP_filteredSVMap[0] = TOP_result_probs_SVM_map_c0;
                TOP_filteredSVMap[1] = TOP_result_probs_SVM_map_c1;
                TOP_filteredSVMap[2] = TOP_result_probs_SVM_map_c2;
                TOP_filteredSVMap[3] = TOP_result_probs_SVM_map_c3;


                int TOP_maxProb = 0;
                [[intel::speculated_iterations(3)]]
                for (int c = 1; c < NUM_CLASSES; c++) {
                    if (TOP_filteredSVMap[c] > TOP_filteredSVMap[TOP_maxProb]) {
                        TOP_maxProb = c;
                    }
                }
                
                labelMap[i] = static_cast<char>(TOP_maxProb + 1);
            }
            
            
        });
    
    });
    
    // BOT WINDOW KNN
    q.submit([&](sycl::handler& h) {
        
        auto pca_accesor = Buff_input_pca_result.get_access<sycl::access::mode::read>(h);
		auto result_probs_SVM = Buff_svmProbabilityMapResult.get_access<sycl::access::mode::read>(h);
		auto labelMap = Buff_labelMap.get_access<sycl::access::mode::write>(h);
    
       h.single_task<VariableWindowKNNBOT>([=]() [[intel::scheduler_target_fmax_mhz(480)]] {
            int winUEdge_corners = pixels_local - safe_border_size_sample;
        
			int winLEdge_corners = HALF_MAX_WS;
            int i_BOT = pixels_local - 1;
            int BOT_zz;
            float SVM_AUX_BOT[MAX_WINDOWSIZE * 4];
            int KNN_AUX[HALF_MAX_WS];
            int j_SVM = NUMBER_OF_PIXELS - MAX_WINDOWSIZE;

            for (int i = 0; i < HALF_MAX_WS; i++) {
                                                
                featureDistance BOT_feat_dist[MAX_WINDOWSIZE];
                
                int nextI = i_BOT - 1;                
                int BOT_pos_feat_dist = 0;
                int BOT_rIdx = (NUMBER_OF_PIXELS - 1 - i) / SAMPLES;
                int BOT_cIdx = (NUMBER_OF_PIXELS - 1 - i) % SAMPLES;
                
                float BOT_PCA_i = pca_accesor[i_BOT];
                
                int loadOffset = 0;
                int jj = MAX_WINDOWSIZE - 1;

                
                for (int j = 0; j < winLEdge_corners; j++) {
                    
//                     if(j < MAX_WINDOWSIZE) {
//                         SVM_AUX_BOT[j*4] = result_probs_SVM[j_SVM*4];
//                         SVM_AUX_BOT[j*4+1] = result_probs_SVM[j_SVM*4+1];
//                         SVM_AUX_BOT[j*4+2] = result_probs_SVM[j_SVM*4+2];
//                         SVM_AUX_BOT[j*4+3] = result_probs_SVM[j_SVM*4+3];
//                         j_SVM++;
//                         loadOffset++;
//                     }
                                                
                    int nextBOT_Pos = BOT_pos_feat_dist + 1;
                    int nextJJ = jj - 1;
                    
                    int BOT_rIdx_j = (NUMBER_OF_PIXELS - 1 - j) / SAMPLES; 
                    int BOT_cIdx_j = (NUMBER_OF_PIXELS - 1 - j) % SAMPLES; 
                    //BOT
                    float BOT_dist = BOT_PCA_i - pca_accesor[jj];
                    int BOT_distr = (BOT_rIdx) - (BOT_rIdx_j); 
                    int BOT_distc = (BOT_cIdx) - (BOT_cIdx_j);                    
                    //BOT                    
                    float BOT_distance = (BOT_dist * BOT_dist) + (BOT_distc * BOT_distc) + (BOT_distr * BOT_distr); 
                    BOT_feat_dist[MAX_WINDOWSIZE - BOT_pos_feat_dist].distance = BOT_distance;
                    BOT_feat_dist[MAX_WINDOWSIZE - BOT_pos_feat_dist].rc = jj;
                    //NEXT ITER
                    BOT_pos_feat_dist = nextBOT_Pos;
                    jj = nextJJ;
                }
                       
                float BOT_min;
				float BOT_last_min = 0;
				int BOT_neighbor[KNN]{};
				int BOT_neighbors[KNN]{};                               
                
                for (int kk = 0; kk < KNN; kk++) {
                    BOT_zz = 0;
                    BOT_min = 1000000.0f;                   
                    
                    for (int ii = MAX_WINDOWSIZE; ii > BOT_pos_feat_dist; ii--) {
                        if ((BOT_feat_dist[ii].distance > BOT_last_min) && (BOT_feat_dist[ii].distance <= BOT_min) && (BOT_feat_dist[ii].distance != 0)) {
							BOT_neighbor[BOT_zz] = BOT_feat_dist[ii].rc;                            
                            if (BOT_min == BOT_feat_dist[ii].distance) {
								BOT_zz++;
							}
							else {
								BOT_zz = 0;
								BOT_min = BOT_feat_dist[ii].distance;
							}
						}
                    }
                    BOT_last_min = BOT_min;
                    for (int x = 0; x <= BOT_zz; x++) {
						if ((kk + x) >= KNN) {
							break;
						}
						BOT_neighbors[kk + x] = BOT_neighbor[x];
					}
					kk += BOT_zz;
                
                }

                winLEdge_corners++;				
                
                float BOT_filteredSVMap[NUM_CLASSES] = {0};                              
                float BOT_result_probs_SVM_map_c0 = 0.0f;
                float BOT_result_probs_SVM_map_c1 = 0.0f;
                float BOT_result_probs_SVM_map_c2 = 0.0f;
                float BOT_result_probs_SVM_map_c3 = 0.0f;

                [[intel::speculated_iterations(20)]]
                for (int z = 0; z < KNN; z = z + 2){
                    int kIdx1 =  BOT_neighbors[z] + 1;
                    int kIdx2 =  BOT_neighbors[z + 1] + 1;
                                    
                    float kidx1_filteredSVMap_c0 = 0.0f;
                    float kidx1_filteredSVMap_c1 = 0.0f;
                    float kidx1_filteredSVMap_c2 = 0.0f;
                    float kidx1_filteredSVMap_c3 = 0.0f;
                    
                    float kidx2_filteredSVMap_c0 = 0.0f;
                    float kidx2_filteredSVMap_c1 = 0.0f;
                    float kidx2_filteredSVMap_c2 = 0.0f;
                    float kidx2_filteredSVMap_c3 = 0.0f;                 
                    
                    
                    if (kIdx1 < pixels_local) {
                        int kIdxOffset1 = kIdx1 * NUM_CLASSES;
//                         kidx1_filteredSVMap_c0 = SVM_AUX_BOT[kIdxOffset1 + 0];
//                         kidx1_filteredSVMap_c1 = SVM_AUX_BOT[kIdxOffset1 + 1];
//                         kidx1_filteredSVMap_c2 = SVM_AUX_BOT[kIdxOffset1 + 2];
//                         kidx1_filteredSVMap_c3 = SVM_AUX_BOT[kIdxOffset1 + 3];                        
                        kidx1_filteredSVMap_c0 = result_probs_SVM[kIdxOffset1 + 0];
                        kidx1_filteredSVMap_c1 = result_probs_SVM[kIdxOffset1 + 1];
                        kidx1_filteredSVMap_c2 = result_probs_SVM[kIdxOffset1 + 2];
                        kidx1_filteredSVMap_c3 = result_probs_SVM[kIdxOffset1 + 3];   
                    }
                    
                    if (kIdx2 < pixels_local) {
                        int kIdxOffset2 = kIdx2 * NUM_CLASSES;
//                         kidx2_filteredSVMap_c0 = SVM_AUX_BOT[kIdxOffset2 + 0];
//                         kidx2_filteredSVMap_c1 = SVM_AUX_BOT[kIdxOffset2 + 1];
//                         kidx2_filteredSVMap_c2 = SVM_AUX_BOT[kIdxOffset2 + 2];
//                         kidx2_filteredSVMap_c3 = SVM_AUX_BOT[kIdxOffset2 + 3];        
                        kidx2_filteredSVMap_c0 = result_probs_SVM[kIdxOffset2 + 0];
                        kidx2_filteredSVMap_c1 = result_probs_SVM[kIdxOffset2 + 1];
                        kidx2_filteredSVMap_c2 = result_probs_SVM[kIdxOffset2 + 2];
                        kidx2_filteredSVMap_c3 = result_probs_SVM[kIdxOffset2 + 3];        
                    }                    


                    BOT_result_probs_SVM_map_c0 += kidx1_filteredSVMap_c0 + kidx2_filteredSVMap_c0;                    
                    BOT_result_probs_SVM_map_c1 += kidx1_filteredSVMap_c1 + kidx2_filteredSVMap_c1;                    
                    BOT_result_probs_SVM_map_c2 += kidx1_filteredSVMap_c2 + kidx2_filteredSVMap_c2;                    
                    BOT_result_probs_SVM_map_c3 += kidx1_filteredSVMap_c3 + kidx2_filteredSVMap_c3;
                }
                
                
                BOT_filteredSVMap[0] = BOT_result_probs_SVM_map_c0;
                BOT_filteredSVMap[1] = BOT_result_probs_SVM_map_c1;
                BOT_filteredSVMap[2] = BOT_result_probs_SVM_map_c2;
                BOT_filteredSVMap[3] = BOT_result_probs_SVM_map_c3;
    

                int BOT_maxProb = 0;
                [[intel::speculated_iterations(3)]]
                for (int c = 1; c < NUM_CLASSES; c++) {
                    if (BOT_filteredSVMap[c] > BOT_filteredSVMap[BOT_maxProb]) {
                        BOT_maxProb = c;
                    }
                }

                labelMap[NUMBER_OF_PIXELS - 1 - i] = static_cast<char>(BOT_maxProb + 1);

                i_BOT = nextI; 
            }                                      
        });
    
    });
    
    // FIXED WINDOW KNN
	q.submit([&](sycl::handler& h) {

		auto pca_accesor = Buff_input_pca_result.get_access<sycl::access::mode::read>(h);
		auto result_probs_SVM = Buff_svmProbabilityMapResult.get_access<sycl::access::mode::read>(h);
		auto labelMap = Buff_labelMap.get_access<sycl::access::mode::write>(h);
                

		h.single_task<FixedWindowKNN>([=]() [[intel::scheduler_target_fmax_mhz(480)]] {                        		            

        int zz;
        int winUEdge = 0;
        int winLEdge = safe_border_size_sample * 2;
        float PCA[NUMBER_OF_PIXELS]; 
	    float SVM_AUX[(NUMBER_OF_PIXELS * 4)];
        constexpr int REDUCED_SIZE = NUMBER_OF_PIXELS - (2 * HALF_MAX_WS);

	    //When using skin cancer HSI, RAM problems may occur and you should make use of the accessors mainly (pca_accesor and result_probs_SVM)
        for (int i = 0; i < NUMBER_OF_PIXELS; i++){
			PCA[i] = pca_accesor[i];
            SVM_AUX[i*4]   = result_probs_SVM[i*4];
            SVM_AUX[i*4+1] = result_probs_SVM[i*4+1];
            SVM_AUX[i*4+2] = result_probs_SVM[i*4+2];
            SVM_AUX[i*4+3] = result_probs_SVM[i*4+3];
		}
        
        for (int i = HALF_MAX_WS; i < NUMBER_OF_PIXELS - HALF_MAX_WS; i++) {
                int rIdx_i = i / SAMPLES;
                int cIdx_i = i % SAMPLES;

                // Accessor Use
                // float PCA_i = pca_accesor[i];

                float PCA_i = PCA[i];

				// Calculations for distance;
                featureDistance feat_dist[HALF_MAX_WS * 2]; // Max size

                int pos_feat_dist = 0;
                
                float min;
                float last_min = 0;
                int neighbor[KNN]{};
                int neighbors[KNN];

                int jj = winUEdge;
                #pragma unroll 12 // correct behavior depends on a common divisor of the number of MAX_WINDOW SIZE and the factor of unroll
                for (int ii = 0; ii < MAX_WINDOWSIZE; ii++) {
                //for (int jj = winUEdge; jj < winLEdge; jj++) {
                    int nextJJ = jj + 1;
                    int rIdx_j = jj / SAMPLES;
                    int cIdx_j = jj % SAMPLES;
                    
                    // Accessor Use
                    // float dist1 = PCA_i - pca_accesor[jj];

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
                    //#pragma unroll 8
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
         
        #pragma unroll 2
        [[intel::speculated_iterations(20)]]
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


            if (kIdx1 < pixels_local) {
                int kIdxOffset1 = kIdx1 * NUM_CLASSES;

                // Accessor use
                // kidx1_filteredSVMap_c0 = result_probs_SVM[kIdxOffset1 + 0];
                // kidx1_filteredSVMap_c1 = result_probs_SVM[kIdxOffset1 + 1];
                // kidx1_filteredSVMap_c2 = result_probs_SVM[kIdxOffset1 + 2];
                // kidx1_filteredSVMap_c3 = result_probs_SVM[kIdxOffset1 + 3];

                kidx1_filteredSVMap_c0 = SVM_AUX[kIdxOffset1 + 0];
                kidx1_filteredSVMap_c1 = SVM_AUX[kIdxOffset1 + 1];
                kidx1_filteredSVMap_c2 = SVM_AUX[kIdxOffset1 + 2];
                kidx1_filteredSVMap_c3 = SVM_AUX[kIdxOffset1 + 3];

            }

            if (kIdx2 < pixels_local) {
                int kIdxOffset2 = kIdx2 * NUM_CLASSES;

                // Accessor use
                // kidx2_filteredSVMap_c0 = result_probs_SVM[kIdxOffset2 + 0];
                // kidx2_filteredSVMap_c1 = result_probs_SVM[kIdxOffset2 + 1];
                // kidx2_filteredSVMap_c2 = result_probs_SVM[kIdxOffset2 + 2];
                // kidx2_filteredSVMap_c3 = result_probs_SVM[kIdxOffset2 + 3];

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



};




std::vector<float> readPcaResultVec(const char* result_filename, int numberOfSamples, int numberOfLines, int numberOfBands) {

	FILE* fp;

	int i, j;
	int np = numberOfLines * numberOfSamples;
	size_t vector_size = static_cast<size_t>(np) * numberOfBands;



	std::vector<float> pcaOneBandResult(vector_size);
	fp = fopen(result_filename, "r");
	if (fp != nullptr) {
		for (i = 0; i < np; i++) {
			for (j = 0; j < numberOfBands; j++) {
				fscanf(fp, "%f", &pcaOneBandResult[i * numberOfBands + j]);
			}
		}
		fclose(fp);
	}
	else {
		cout << "Error Reading File: " << result_filename << std::endl;
	}
	

	return pcaOneBandResult;
}




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
	//auto selector = sycl::ext::intel::fpga_emulator_selector_v;
	sycl::queue q(selector);


	auto device = q.get_device();
	std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;

//	if (!device.has(sycl::aspect::usm_host_allocations)) std::terminate();


	//clock_t startPCA, endPCA, startSVM, endSVM, startKNN, endKNN;
	int numberOfLines = 442;
	int numberOfSamples = 496;
	//int numberOfBands = 826;
	int numberOfBands = 128; //after the pre-processing brain = 128, derma = 100;
	int numberOfPixels = (numberOfLines * numberOfSamples);

	//PCA DATA
	std::vector<float> pcaOneBandResult;//(NUMBER_OF_PIXELS * NUMBER_OF_BANDS);
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

	
	//std::string pcaOutput = "./sections/PCA/Output_PCA_unidimensional.txt";
	//std::fstream fp_pca;
	//fp_pca.open(pcaOutput, std::fstream::out | std::fstream::trunc);

	//for (int i = 0; i < numberOfPixels; i++) {
	//	for (int j = 0; j < numberOfPcaBands; j++) {
	//		fp_pca << std::setprecision(6) << std::fixed << pcaOneBandResult[i * numberOfPcaBands + j] << "\t";
	//		/*if(i < 1) cout << std::setprecision(6) << std::fixed << pcaOneBandResult[i * numberOfPcaBands + j] << "\n";*/
	//		//fprintf(fp_pca, "%lf\t", pcaOneBandResult[i][j]);
	//	}
	//	//fprintf(fp_pca, "\n");
	//	fp_pca << "\n";
	//}

	//fp_pca.close();



	printf("Start SVM read...\n");

	//std::string svmOutput = "Output_SVM_fgpaCompile.txt";
	//std::string svmOutput = "Output_SVM.txt";
	std::string svmOutput = "./sections/SVM/Output_SVM_UnidimensialArray.txt";

	//startSVM = clock();
	svmProbabilityMapResult = readSVMResultVec("Output_SVM.txt", numberOfLines, numberOfSamples, numberOfBands, numberOfClasses);
	//endSVM = clock();

	//std::cout << "Time SVM\t\t\t\t--->" << std::setprecision(5) << executionTime(startSVM, endSVM) << "seconds \n";
	// ---- PRINT SVM OUTPUT----
	
	/*std::fstream fp_svm;
	fp_svm.open(svmOutput, std::fstream::out | std::fstream::trunc);
	
	for (int i = 0; i < numberOfPixels; i++) {
		for (int j = 0; j < numberOfClasses; j++) {
			fp_svm << std::setprecision(6) << std::fixed << svmProbabilityMapResult[i*numberOfClasses + j] << "\t";
		}
		fp_svm << "\n";
	}
	fp_svm.close();*/

	printf("Start kNN algo...\n");
	//startKNN = clock();
    auto start = std::chrono::high_resolution_clock::now();

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
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //endKNN = clock();	
	//std::cout << "Time KNN\t\t\t\t--->" << std::setprecision(5) << executionTime(startKNN, endKNN) << "seconds \n";
    std::cout << "KNNkernel execution time: " << duration.count() << " milliseconds" << std::endl;

	std::string knnOutput = "./sections/KNN/Output_KNN_fgpaCompile.txt";
	//std::string knnOutput = "Output_KNN.txt";

	std::fstream fp_knn;
	
	fp_knn.open(knnOutput, std::fstream::out | std::fstream::trunc);
	for (int i = 0; i < numberOfPixels; i++) {
		fp_knn << (int)knnFilteredMap[i] << " \n";		
	}
	fp_knn.close();
	
	if(f_time.is_open())
		f_time.close();

	return 0;
}

