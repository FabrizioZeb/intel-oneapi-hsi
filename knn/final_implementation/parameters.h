#pragma once

/*CONSTANTS*/

#define SAMPLES 496
#define NUMBER_OF_LINES 442
//#define NUMBER_OF_BANDS 826
#define NUMBER_OF_BANDS 128
#define NUMBER_OF_PIXELS 219232 //int numberOfPixels = (numberOfLines * numberOfSamples);

/*PCA CONSTANT*/
#define OPTION 0

/*SVM Constants*/
//#define BANDS 100
#define NUM_CLASSES 4
#define NUM_BINARY_CLASSIFIERS ((NUM_CLASSES * (NUM_CLASSES -1))/2)

/*KNN Constants*/
#define WINDOWSIZE 6 //6 // 8 /*Mejor*/                //12 //14
#define SAFEBORDERSIZE (WINDOWSIZE/2)
#define KNN 40
#define LAMBDA 1

#define MAX_WINDOWSIZE (SAFEBORDERSIZE * SAMPLES * 2)
#define HALF_MAX_WS (SAFEBORDERSIZE * SAMPLES)
#define AUX_SVM_WS (MAX_WINDOWSIZE * 4)
#define KNN_BOT_OFFSET (NUMBER_OF_PIXELS - HALF_MAX_WS)


#define SVM_FIX_WS ((NUMBER_OF_PIXELS - (HALF_MAX_WS * 2)) * 4)
//KNN Structures
typedef struct featureMatrixNode {
	//double PCA_pVal;
	float PCA_pVal;
	short r;
	short c;
	int rc;
} featureMatrixMode;

typedef struct featureDistance {
	//double distance;
	float distance;
	int rc;
} featureDistance;