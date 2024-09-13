
/*PCA COSTANTS*/
#define OPTION 0

/*SVM COSTANTS*/
//#define BANDS 100
#define NUM_CLASSES		4 
#define	NUM_BINARY_CLASSIFIERS	((NUM_CLASSES * (NUM_CLASSES-1))/2)

/*KNN COSTANTS*/
#define WINDOWSIZE					12 //14
#define SAFEBORDERSIZE				(WINDOWSIZE/2)
#define KNN							40
#define LAMBDA						1
//KNN structures
typedef struct featureMatrixNode {
	double PCA_pVal;
	short r;
	short c;
	int rc;
}featureMatrixNode;

typedef struct featureDistance {
	double distance;
	int rc;
}featureDistance;