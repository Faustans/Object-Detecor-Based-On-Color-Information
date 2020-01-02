#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/photo/cuda.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace cv::ximgproc::segmentation;
using namespace std;

RNG rng(12345);
int thresh = 100;
int max_thresh = 255;
const Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));


int main(int argc, char** argv) {
	//for (int i = 0; i<=argc;i++){
//		cout << argv[i] << endl;
//	}
	VideoCapture cap;
	cap.set(CAP_PROP_FRAME_WIDTH, 500);
	cap.set(CAP_PROP_FRAME_HEIGHT, 600);
	if( argc == 2)
	{
		string arg1(argv[1]);
		if (arg1.compare("-c") == 0)
		{
			std::cout << "Opening with camera" << std::endl;
			cap=VideoCapture(0);
		}
		else
		{
			std::cout << "Error reading arg" << std::endl;
			return -1;
		}
	}
	else if ( argc == 3)
	{
		string arg1(argv[1]);
		if (arg1.compare("-f") == 0)
		{
			printf("Opening video %s\n", argv[2]);
			cap=VideoCapture(argv[2]);
		}
		else
		{
			std::cout << "Error reading arg" << std::endl;
			return -1;
		}
	}
	else
	{
		std::cout << "Don't know what to read" << std::endl;
		return 0;
	}

	// load video
	if (!cap.isOpened())
	{
		cout << "Error at opening video" << std::endl;
		return -1;
	}
	Mat background;
	for(int i=0;i<10;i++)
	{
		cap >> background;
	}

	//Laterally invert the image / flip the image.
	flip(background,background,1);
	while(1){
    Mat frame;
		Mat treated_output;

    // Capture frame-by-frame
    cap >> frame;

		// inverting frame
		flip(frame,frame,1);

		//Converting image from BGR to HSV color space.
		Mat hsv;
		cvtColor(frame, hsv, COLOR_BGR2HSV);

		Mat mask1, mask2;

		// mask for black
		//inRange(hsv, Scalar(0, 0, 0), Scalar(180,255,30), mask1);
		// mask for red
		//inRange(hsv, Scalar(0, 120, 150), Scalar(10, 255, 255), mask1);
		//inRange(hsv, Scalar(170, 120, 120), Scalar(180, 255, 255), mask1); //Red
		//inRange(hsv, 	Scalar(0, 0, 0, 0), Scalar(180, 255, 30, 0), mask1); Black
		//inRange(hsv, 	Scalar(0, 0, 200, 0), Scalar(180, 255, 255, 0), mask1); White

		inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
		inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);

		//inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
    //inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);

		// Generating the final mask
		mask1 = mask1 + mask2;

		Mat kernel = Mat::ones(3,3, CV_32F);
		morphologyEx(mask1,mask1,cv::MORPH_OPEN,kernel);
		morphologyEx(mask1,mask1,cv::MORPH_DILATE,kernel);

		// creating an inverted mask to segment out the cloth from the frame
		bitwise_not(mask1,mask2);
		Mat res1, res2, final_output;


		// Segmenting the cloth out of the frame using bitwise and with the inverted mask
		bitwise_and(frame,frame,res1,mask2);

    // If the frame is empty, break immediately
		Mat blank = Mat( background.size(), CV_8UC3, Scalar(255,255,255) );
		bitwise_and(blank,blank,res2,mask1);

		// Generating the final augmented output.
		addWeighted(res1,1,res2,1,0,final_output);

		//imshow("res1", res1);
		//imshow("res2", res2);
		//drawing square around object
		Mat denoised;
		Mat cnts;
  	vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		//cvtColor(res2, res2, COLOR_BGR2GRAY, 1);
		//fastNlMeansDenoising(res2, denoised,1, 7, 21);
		Canny( res2, cnts, thresh, thresh*2, 3 );

		findContours(cnts,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

		//Try to draw Square around color object;

		Mat drawing = Mat::zeros( cnts.size(), CV_8UC3 );
		double area = 0;
		int x = 0;
	 	for( int i = 0; i< contours.size(); i++ )
	     {
	       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				 if(contourArea(contours[i])>area){
					 area = contourArea(contours[i]);
					 x = i;
				 }
				 //Mat x;
				 //boxPoints(r,x);
	       //drawContours( drawing, x, i, color, 2, 8, hierarchy, 0, Point() );
		}
		if(contours.size()>0){
			Mat obj_area;
 		 //double ep = 0.1*arcLength(contours[x],true);
 		 //approxPolyDP(contours[x],obj_area,ep,true);
 		 Rect r = boundingRect(contours[x]);
 		 rectangle(frame,r,Scalar(0,255,0),1,8,0);
		}

		imshow("magic", frame);
		imshow("drawing", drawing);
    if (frame.empty())
      break;

    // Display the resulting frame
    //imshow( "HSV", hsv );
		//imshow( "Frame", frame);

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }
  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  destroyAllWindows();

	return 0;
}
