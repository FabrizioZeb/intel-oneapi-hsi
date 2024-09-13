#define OPTION 0
#define NUM_SAMPLES 496 //496
#define NUM_LINES 442 //442
#define NUM_BANDS 128 //128
#define NUM_PCA_BANDS 1 //1
#define NUM_PIXELS 219232 //NUM_SAMPLES * NUM_LINES

//SVM CONSTANTS
#define NUM_CLASSES 4
#define NUM_BINARY_CLASSIFIERS ((NUM_CLASSES * (NUM_CLASSES -1))/2)

//KNN CONSTANTS
#define WINDOWSIZE 6 //6 // 8 /*Mejor*/                //12 //14
#define SAFEBORDERSIZE (WINDOWSIZE/2)
#define KNN 40
#define LAMBDA 1

#define MAX_WINDOWSIZE (SAFEBORDERSIZE * NUM_SAMPLES * 2)
#define HALF_MAX_WS (SAFEBORDERSIZE * NUM_SAMPLES)
#define AUX_SVM_WS (MAX_WINDOWSIZE * 4)
#define KNN_BOT_OFFSET (NUM_PIXELS - HALF_MAX_WS)

#define SVM_FIX_WS ((NUM_PIXELS - (HALF_MAX_WS * 2)) * 4)
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