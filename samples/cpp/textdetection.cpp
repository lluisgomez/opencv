/*
 * textdetection.cpp
 *
 * A demo program of the Extremal Region Filter algorithm described in
 * Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
 *
 * Created on: Sep 23, 2013
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

#include  "opencv2/opencv.hpp"
#include  "opencv2/objdetect.hpp"
#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"

#include  <vector>
#include  <iostream>
#include  <iomanip>

using  namespace std;
using  namespace cv;

void show_help_and_exit(const char *cmd);
void groups_draw(Mat &src, vector<Rect> &groups);
void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions);

//This to be moved to include file

//these threshold values are learned from training dataset
#define PAIR_MIN_HEIGHT_RATIO    0.4
#define PAIR_MIN_CENTROID_ANGLE -0.85
#define PAIR_MAX_CENTROID_ANGLE  0.85
#define PAIR_MIN_REGION_DIST    -0.4 
#define PAIR_MAX_REGION_DIST     2.2

#define TRIPLET_MAX_DIST         0.9
#define TRIPLET_MAX_SLOPE        0.3


struct line_estimates
{
  float top1_a0;
  float top1_a1;
  float top2_a0;
  float top2_a1;
  float bottom1_a0;
  float bottom1_a1;
  float bottom2_a0;
  float bottom2_a1;
  int x_min;
  int x_max;
  int h_max;
};

float distanceLinesEstimates(line_estimates &a, line_estimates &b);

float distanceLinesEstimates(line_estimates &a, line_estimates &b)
{
  int x_min = min(a.x_min, b.x_min);
  int x_max = max(a.x_max, b.x_max);
  int h_max = max(a.h_max, b.h_max);

  float dist_top = INT_MAX, dist_bottom = INT_MAX;
  for (int i=0; i<2; i++)
  {
    float top_a0, top_a1, bottom_a0, bottom_a1;
    if (i == 0)
    {
      top_a0 = a.top1_a0;
      top_a1 = a.top1_a1;
      bottom_a0 = a.bottom1_a0;
      bottom_a1 = a.bottom1_a1;
    } else {
      top_a0 = a.top2_a0;
      top_a1 = a.top2_a1;
      bottom_a0 = a.bottom2_a0;
      bottom_a1 = a.bottom2_a1;
    }
    for (int j=0; j<2; j++)
    {
      float top_b0, top_b1;
      float bottom_b0, bottom_b1;
      if (j==0)
      {
        top_b0 = b.top1_a0;
        top_b1 = b.top1_a1;
        bottom_b0 = b.bottom1_a0;
        bottom_b1 = b.bottom1_a1;
      } else {
        top_b0 = b.top2_a0;
        top_b1 = b.top2_a1;
        bottom_b0 = b.bottom2_a0;
        bottom_b1 = b.bottom2_a1;
      }

      float x_min_dist = abs((top_a0+x_min*top_a1) - (top_b0+x_min*top_b1));
      float x_max_dist = abs((top_a0+x_max*top_a1) - (top_b0+x_max*top_b1));
      dist_top    = min(dist_top, max(x_min_dist,x_max_dist)/h_max);

      x_min_dist  = abs((bottom_a0+x_min*bottom_a1) - (bottom_b0+x_min*bottom_b1));
      x_max_dist  = abs((bottom_a0+x_max*bottom_a1) - (bottom_b0+x_max*bottom_b1));
      dist_bottom = min(dist_bottom, max(x_min_dist,x_max_dist)/h_max);
    }
  }

  return max(dist_top, dist_bottom);
}

struct region_pair
{
	Vec2i a;
	Vec2i b;
  region_pair (Vec2i _a, Vec2i _b) : a(_a), b(_b) {}
};

struct region_triplet
{
	Vec2i a;
	Vec2i b;
	Vec2i c;
  line_estimates estimates;
  region_triplet (Vec2i _a, Vec2i _b, Vec2i _c) : a(_a), b(_b), c(_c) {}
};

bool isValidPair(std::vector< std::vector<ERStat> >& regions, cv::Vec2i idx1, cv::Vec2i idx2);
bool isValidTriplet(std::vector< std::vector<ERStat> >& regions, region_pair pair1, region_pair pair2, region_triplet &triplet);
void erGroupingNM(cv::InputArrayOfArrays _src, std::vector< std::vector<ERStat> >& regions,  std::vector< std::vector<Vec2i> >& groups);
// Fit line from two points
// out a0 is the intercept
// out a1 is the slope
void fitLine(Point p1, Point p2, float &a0, float &a1);

// Fit line from three points using Ordinary Least Squares
// out a0 is the intercept
// out a1 is the slope
void fitLineOLS(Point p1, Point p2, Point p3, float &a0, float &a1);

// Fit line from three points using (heuristic) Least-Median of Squares
// out a0 is the intercept
// out a1 is the slope
// returns the error of the single point that doesn't fit the line
float fitLineLMS(Point p1, Point p2, Point p3, float &a0, float &a1);

void fitLineEstimates(vector< vector<ERStat> > &regions, region_triplet &triplet);

// Fit line from two points
// out a0 is the intercept
// out a1 is the slope
void fitLine(Point p1, Point p2, float &a0, float &a1)
{
  CV_Assert ( p1.x != p2.x );

  a1 = (float)(p2.y - p1.y) / (p2.x - p1.x);
  a0 = a1 * -1 * p1.x + p1.y;
}

// Fit line from three points using Ordinary Least Squares
// out a0 is the intercept
// out a1 is the slope
void fitLineOLS(Point p1, Point p2, Point p3, float &a0, float &a1)
{
  float sumx  = p1.x + p2.x + p3.x;
  float sumy  = p1.y + p2.y + p3.y;
  float sumxy = p1.x*p1.y + p2.x*p2.y + p3.x*p3.y;
  float sumx2 = p1.x*p1.x + p2.x*p2.x + p3.x*p3.x;

  // line coefficients
  a0=(float)(sumy*sumx2-sumx*sumxy) / (3*sumx2-sumx*sumx);
  a1=(float)(3*sumxy-sumx*sumy) / (3*sumx2-sumx*sumx);
}

// Fit line from three points using (heutistic) Least-Median of Squares
// out a0 is the intercept
// out a1 is the slope
// returns the error of the single point that doesn't fit the line
float fitLineLMS(Point p1, Point p2, Point p3, float &a0, float &a1)
{

  //Least-Median of Squares does not make sense with only three points
  //becuse any line passing by two of them has median_error = 0
  //So we'll take the one with smaller slope
  float l_a0, l_a1, best_slope, err;

  fitLine(p1,p2,a0,a1);
  best_slope = abs(a1);
  err = (p3.y - (a0+a1*p3.x));

  fitLine(p1,p3,l_a0,l_a1);
  if (abs(l_a1) < best_slope)
  {
    best_slope = abs(l_a1);
    a0 = l_a0;
    a1 = l_a1;
    err = (p2.y - (a0+a1*p2.x));
  }

  fitLine(p2,p3,l_a0,l_a1);
  if (abs(l_a1) < best_slope)
  {
    best_slope = abs(l_a1);
    a0 = l_a0;
    a1 = l_a1;
    err = (p1.y - (a0+a1*p1.x));
  }
  return err;

}

void fitLineEstimates(vector< vector<ERStat> > &regions, region_triplet &triplet)
{
  vector<Rect> char_boxes;
  char_boxes.push_back(regions[triplet.a[0]][triplet.a[1]].rect);
  char_boxes.push_back(regions[triplet.b[0]][triplet.b[1]].rect);
  char_boxes.push_back(regions[triplet.c[0]][triplet.c[1]].rect);

  triplet.estimates.x_min = min(min(char_boxes[0].tl().x,char_boxes[1].tl().x), char_boxes[2].tl().x);
  triplet.estimates.x_max = max(max(char_boxes[0].br().x,char_boxes[1].br().x), char_boxes[2].br().x);
  triplet.estimates.h_max = max(max(char_boxes[0].height,char_boxes[1].height), char_boxes[2].height);

  // Fit one bottom line
  float err = fitLineLMS(char_boxes[0].br(), char_boxes[1].br(), char_boxes[2].br(), 
                         triplet.estimates.bottom1_a0, triplet.estimates.bottom1_a1);

  // Slope for all lines is the same
  triplet.estimates.bottom2_a1 = triplet.estimates.bottom1_a1;
  triplet.estimates.top1_a1    = triplet.estimates.bottom1_a1;
  triplet.estimates.top2_a1    = triplet.estimates.bottom1_a1;

  if (abs(err) > (float)triplet.estimates.h_max/6)
  {
    // We need two different bottom lines
    triplet.estimates.bottom2_a0 = triplet.estimates.bottom1_a0 + err;
  }
  else 
  {
    // Second bottom line is the same
    triplet.estimates.bottom2_a0 = triplet.estimates.bottom1_a0;
  }

  // Fit one top line within the two (Y)-closer coordinates
  int d_12 = abs(char_boxes[0].tl().y - char_boxes[1].tl().y);
  int d_13 = abs(char_boxes[0].tl().y - char_boxes[2].tl().y);
  int d_23 = abs(char_boxes[1].tl().y - char_boxes[2].tl().y);
  if ((d_12<d_13) && (d_12<d_23))
  {
    Point p = Point((char_boxes[0].tl().x + char_boxes[1].tl().x)/2, 
                    (char_boxes[0].tl().y + char_boxes[1].tl().y)/2);
    triplet.estimates.top1_a0 = triplet.estimates.bottom1_a0 + 
                                (p.y - (triplet.estimates.bottom1_a0+p.x*triplet.estimates.bottom1_a1));
    p = char_boxes[2].tl();
    err = (p.y - (triplet.estimates.top1_a0+p.x*triplet.estimates.top1_a1));
  }
  else if (d_13<d_23)
  {
    Point p = Point((char_boxes[0].tl().x + char_boxes[2].tl().x)/2, 
                    (char_boxes[0].tl().y + char_boxes[2].tl().y)/2);
    triplet.estimates.top1_a0 = triplet.estimates.bottom1_a0 + 
                                (p.y - (triplet.estimates.bottom1_a0+p.x*triplet.estimates.bottom1_a1));
    p = char_boxes[1].tl();
    err = (p.y - (triplet.estimates.top1_a0+p.x*triplet.estimates.top1_a1));
  }
  else
  {
    Point p = Point((char_boxes[1].tl().x + char_boxes[2].tl().x)/2, 
                    (char_boxes[1].tl().y + char_boxes[2].tl().y)/2);
    triplet.estimates.top1_a0 = triplet.estimates.bottom1_a0 + 
                                (p.y - (triplet.estimates.bottom1_a0+p.x*triplet.estimates.bottom1_a1));
    p = char_boxes[0].tl();
    err = (p.y - (triplet.estimates.top1_a0+p.x*triplet.estimates.top1_a1));
  }

  if (abs(err) > (float)triplet.estimates.h_max/6)
  {
    // We need two different top lines
    triplet.estimates.top2_a0 = triplet.estimates.top1_a0 + err;
  }
  else 
  {
    // Second top line is the same
    triplet.estimates.top2_a0 = triplet.estimates.top1_a0;
  }
}


bool isValidPair(std::vector< std::vector<ERStat> >& regions, cv::Vec2i idx1, cv::Vec2i idx2)
{
	Rect minarearect  = regions[idx1[0]][idx1[1]].rect | regions[idx2[0]][idx2[1]].rect;

	if ( (minarearect == regions[idx1[0]][idx1[1]].rect) || (minarearect == regions[idx2[0]][idx2[1]].rect) )
		return false;
        
  ERStat *i, *j;
  if (regions[idx1[0]][idx1[1]].rect.x < regions[idx2[0]][idx2[1]].rect.x)
  {
    i = &regions[idx1[0]][idx1[1]];
    j = &regions[idx2[0]][idx2[1]];
  } else {
    i = &regions[idx2[0]][idx2[1]];
    j = &regions[idx1[0]][idx1[1]];
  }
  
  if (j->rect.x == i->rect.x)
    return false;
    
  float height_ratio = (float)min(i->rect.height,j->rect.height) /
                              max(i->rect.height,j->rect.height);
        
  Point center_i(i->rect.x+i->rect.width/2, i->rect.y+i->rect.height/2);
  Point center_j(j->rect.x+j->rect.width/2, j->rect.y+j->rect.height/2);
  float centroid_angle = atan2(center_j.y-center_i.y, center_j.x-center_i.x);
        
  int avg_width = (i->rect.width + j->rect.width) / 2;
  float norm_distance = (float)(j->rect.x-(i->rect.x+i->rect.width))/avg_width;

	if (( height_ratio   < PAIR_MIN_HEIGHT_RATIO) ||
      ( centroid_angle < PAIR_MIN_CENTROID_ANGLE) ||
      ( centroid_angle > PAIR_MAX_CENTROID_ANGLE) ||
      ( norm_distance  < PAIR_MIN_REGION_DIST) ||
      ( norm_distance  > PAIR_MAX_REGION_DIST))
		return false;

	return true;
}

bool isValidTriplet(std::vector< std::vector<ERStat> >& regions, region_pair pair1, region_pair pair2, region_triplet &triplet)
{
	if ( (pair1.a == pair2.a)||(pair1.a == pair2.b)||(pair1.b == pair2.a)||(pair1.b == pair2.b) )
	{
		//it's a possible triplet, now check if it's valid
		if (true)
		{
      //fill the indexes in the output tripled (sorted)
	    if (pair1.a == pair2.a)
      {
        if ((regions[pair1.b[0]][pair1.b[1]].rect.x <= regions[pair1.a[0]][pair1.a[1]].rect.x) &&
            (regions[pair2.b[0]][pair2.b[1]].rect.x <= regions[pair1.a[0]][pair1.a[1]].rect.x))
           return false;
        if ((regions[pair1.b[0]][pair1.b[1]].rect.x >= regions[pair1.a[0]][pair1.a[1]].rect.x) &&
            (regions[pair2.b[0]][pair2.b[1]].rect.x >= regions[pair1.a[0]][pair1.a[1]].rect.x))
           return false;

        triplet.a = (regions[pair1.b[0]][pair1.b[1]].rect.x < regions[pair2.b[0]][pair2.b[1]].rect.x)? pair1.b : pair2.b;
        triplet.b = pair1.a;
        triplet.c = (regions[pair1.b[0]][pair1.b[1]].rect.x > regions[pair2.b[0]][pair2.b[1]].rect.x)? pair1.b : pair2.b;

      } else if (pair1.a == pair2.b) {
        if ((regions[pair1.b[0]][pair1.b[1]].rect.x <= regions[pair1.a[0]][pair1.a[1]].rect.x) &&
            (regions[pair2.a[0]][pair2.a[1]].rect.x <= regions[pair1.a[0]][pair1.a[1]].rect.x))
           return false;
        if ((regions[pair1.b[0]][pair1.b[1]].rect.x >= regions[pair1.a[0]][pair1.a[1]].rect.x) &&
            (regions[pair2.a[0]][pair2.a[1]].rect.x >= regions[pair1.a[0]][pair1.a[1]].rect.x))
           return false;

        triplet.a = (regions[pair1.b[0]][pair1.b[1]].rect.x < regions[pair2.a[0]][pair2.a[1]].rect.x)? pair1.b : pair2.a;
        triplet.b = pair1.a;
        triplet.c = (regions[pair1.b[0]][pair1.b[1]].rect.x > regions[pair2.a[0]][pair2.a[1]].rect.x)? pair1.b : pair2.a;

      } else if (pair1.b == pair2.a) {
        if ((regions[pair1.a[0]][pair1.a[1]].rect.x <= regions[pair1.b[0]][pair1.b[1]].rect.x) &&
            (regions[pair2.b[0]][pair2.b[1]].rect.x <= regions[pair1.b[0]][pair1.b[1]].rect.x))
           return false;
        if ((regions[pair1.a[0]][pair1.a[1]].rect.x >= regions[pair1.b[0]][pair1.b[1]].rect.x) &&
            (regions[pair2.b[0]][pair2.b[1]].rect.x >= regions[pair1.b[0]][pair1.b[1]].rect.x))
           return false;

        triplet.a = (regions[pair1.a[0]][pair1.a[1]].rect.x < regions[pair2.b[0]][pair2.b[1]].rect.x)? pair1.a : pair2.b;
        triplet.b = pair1.b;
        triplet.c = (regions[pair1.a[0]][pair1.a[1]].rect.x > regions[pair2.b[0]][pair2.b[1]].rect.x)? pair1.a : pair2.b;

      } else if (pair1.b == pair2.b) {
        if ((regions[pair1.a[0]][pair1.a[1]].rect.x <= regions[pair1.b[0]][pair1.b[1]].rect.x) &&
            (regions[pair2.a[0]][pair2.a[1]].rect.x <= regions[pair1.b[0]][pair1.b[1]].rect.x))
           return false;
        if ((regions[pair1.a[0]][pair1.a[1]].rect.x >= regions[pair1.b[0]][pair1.b[1]].rect.x) &&
            (regions[pair2.a[0]][pair2.a[1]].rect.x >= regions[pair1.b[0]][pair1.b[1]].rect.x))
           return false;

        triplet.a = (regions[pair1.a[0]][pair1.a[1]].rect.x < regions[pair2.a[0]][pair2.a[1]].rect.x)? pair1.a : pair2.a;
        triplet.b = pair1.b;
        triplet.c = (regions[pair1.a[0]][pair1.a[1]].rect.x > regions[pair2.a[0]][pair2.a[1]].rect.x)? pair1.a : pair2.a;

      }

		}

    fitLineEstimates(regions, triplet);

    if ( (triplet.estimates.bottom1_a0 < triplet.estimates.top1_a0) || 
         (triplet.estimates.bottom1_a0 < triplet.estimates.top2_a0) || 
         (triplet.estimates.bottom2_a0 < triplet.estimates.top1_a0) || 
         (triplet.estimates.bottom2_a0 < triplet.estimates.top2_a0) )
      return false; 

    int central_height = min(triplet.estimates.bottom1_a0, triplet.estimates.bottom2_a0) -
                         max(triplet.estimates.top1_a0,triplet.estimates.top2_a0);
    int top_height     = abs(triplet.estimates.top1_a0 - triplet.estimates.top2_a0);
    int bottom_height  = abs(triplet.estimates.bottom1_a0 - triplet.estimates.bottom2_a0);

    float top_height_ratio    = (float)top_height/central_height;
    float bottom_height_ratio = (float)bottom_height/central_height;

    if ( (top_height_ratio > TRIPLET_MAX_DIST) || (bottom_height_ratio > TRIPLET_MAX_DIST) )
      return false;

    if (abs(triplet.estimates.bottom1_a1) > TRIPLET_MAX_SLOPE)
      return false;

		return true;
	}

	return false;
}

void erGroupingNM(cv::InputArrayOfArrays _src, std::vector< std::vector<ERStat> >& regions,  std::vector< std::vector<Vec2i> >& groups)
{

  std::vector<Mat> src;
  _src.getMatVector(src);

  CV_Assert ( !src.empty() );
  CV_Assert ( src.size() == regions.size() );

  size_t num_channels = src.size();
	
	std::vector< cv::Vec2i > all_regions;
	std::vector< region_pair > valid_pairs;

	//store indices to regions in a single vector
	for(size_t c=0; c<num_channels; c++)
	{
		for(size_t r=0; r<regions[c].size(); r++)
		{
			all_regions.push_back(Vec2i(c,r));
		}
	}

	//check every possible pair of regions
	for (size_t i=0; i<all_regions.size(); i++)
	{
		for (size_t j=i+1; j<all_regions.size(); j++)
		{
			// check height ratio, centroid angle and region distance normalized by region width fall within a given interval
			if (isValidPair(regions, all_regions[i],all_regions[j]))
			{
				valid_pairs.push_back(region_pair(all_regions[i],all_regions[j]));
			}
		}
	}

  cout << "GroupingNM : detected " << valid_pairs.size() << " valid pairs" << endl;

	std::vector< region_triplet > valid_triplets;

	//check every possible triplet of regions
	for (size_t i=0; i<valid_pairs.size(); i++)
	{
		for (size_t j=i+1; j<valid_pairs.size(); j++)
		{
			// check colinearity rule
			region_triplet valid_triplet(Vec2i(0,0),Vec2i(0,0),Vec2i(0,0));
			if (isValidTriplet(regions, valid_pairs[i],valid_pairs[j], valid_triplet))
			{
				valid_triplets.push_back(valid_triplet);
			}
		}
	}
	
  cout << "GroupingNM : detected " << valid_triplets.size() << " valid triplets" << endl;

	//TODO remove this, it is only to visualize 
  Mat lines = Mat::zeros(src[0].rows+2,src[0].cols+2,CV_8UC3);
	for (size_t i=0; i<valid_triplets.size(); i++)
	{
    ERStat *a,*b,*c;
    a = &regions[valid_triplets[i].a[0]][valid_triplets[i].a[1]];
    b = &regions[valid_triplets[i].b[0]][valid_triplets[i].b[1]];
    c = &regions[valid_triplets[i].c[0]][valid_triplets[i].c[1]];
    Point center_a(a->rect.x+a->rect.width/2, a->rect.y+a->rect.height/2);
    Point center_b(b->rect.x+b->rect.width/2, b->rect.y+b->rect.height/2);
    Point center_c(c->rect.x+c->rect.width/2, c->rect.y+c->rect.height/2);

    line(lines,center_a,center_b, Scalar(0,0,255),2);
    line(lines,center_b,center_c, Scalar(0,0,255),2);
/*
          Mat drawing = Mat::zeros( lines.size(), CV_8UC3);

          rectangle(drawing, a->rect.tl(), a->rect.br(), Scalar(255,0,0));
          rectangle(drawing, b->rect.tl(), b->rect.br(), Scalar(255,0,0));
          rectangle(drawing, c->rect.tl(), c->rect.br(), Scalar(255,0,0));

          line(drawing,center_a,center_b, Scalar(0,0,255));
          line(drawing,center_b,center_c, Scalar(0,0,255));

          line(drawing, Point(0,(int)valid_triplets[i].estimates.bottom1_a0), 
                        Point(drawing.cols,(int)(valid_triplets[i].estimates.bottom1_a0+valid_triplets[i].estimates.bottom1_a1*drawing.cols)), Scalar(0,255,0));
          line(drawing, Point(0,(int)valid_triplets[i].estimates.bottom2_a0), 
                        Point(drawing.cols,(int)(valid_triplets[i].estimates.bottom2_a0+valid_triplets[i].estimates.bottom2_a1*drawing.cols)), Scalar(0,255,0));
          line(drawing, Point(0,(int)valid_triplets[i].estimates.top1_a0), 
                        Point(drawing.cols,(int)(valid_triplets[i].estimates.top1_a0+valid_triplets[i].estimates.top1_a1*drawing.cols)), Scalar(0,255,0));
          line(drawing, Point(0,(int)valid_triplets[i].estimates.top2_a0), 
                        Point(drawing.cols,(int)(valid_triplets[i].estimates.top2_a0+valid_triplets[i].estimates.top2_a1*drawing.cols)), Scalar(0,255,0));

          imshow( "line estimates", drawing );
          waitKey(0);*/

  }
  imshow("lines",lines);
	
}

int  main(int argc, const char * argv[])
{

    cout << endl << argv[0] << endl << endl;
    cout << "Demo program of the Extremal Region Filter algorithm described in " << endl;
    cout << "Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012" << endl << endl;

    if (argc < 2) show_help_and_exit(argv[0]);

    Mat src = imread(argv[1]);

    // Extract channels to be processed individually
    vector<Mat> channels;
    computeNMChannels(src, channels);

    int cn = (int)channels.size();
    // Append negative channels to detect ER- (bright regions over dark background)
    for (int c = 0; c < cn-1; c++)
        channels.push_back(255-channels[c]);

    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),8,0.00015,0.13,0.2,true,0.1);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);

    vector<vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    cout << "Extracting Class Specific Extremal Regions from " << (int)channels.size() << " channels ..." << endl;
    cout << "    (...) this may take a while (...)" << endl << endl;
    int num_regions = 0;
    for (int c=0; c<(int)channels.size(); c++)
    {
	    	double t = (double)cvGetTickCount();
        er_filter1->run(channels[c], regions[c]);
	    	t = cvGetTickCount() - t;
	    	printf( "filter 1 done in %g ms.\n", t/((double)cvGetTickFrequency()*1000.) );
	    	t = (double)cvGetTickCount();
        er_filter2->run(channels[c], regions[c]);
	    	t = cvGetTickCount() - t;
	    	printf( "filter 2 done in %g ms.\n", t/((double)cvGetTickFrequency()*1000.) );
        num_regions += regions[c].size();
    }
    cout << "In total we have " << num_regions << endl;

    // Detect character groups
    cout << "Grouping extracted ERs ... ";
    vector<Rect> groups;
	  double t = (double)cvGetTickCount();
    erGrouping(channels, regions, "trained_classifier_erGrouping.xml", 0.5, groups);
	  t = cvGetTickCount() - t;
	  printf( "Grouping done in %g ms.\n", t/((double)cvGetTickFrequency()*1000.) );

    // draw groups
    groups_draw(src, groups);
    imshow("grouping",src);

    // Groouping using Exhaustive Search algorithm
    cout << "GroupingNM extracted ERs ... " << endl;
    vector< vector<Vec2i> > nm_groups;
	  t = (double)cvGetTickCount();
    erGroupingNM(channels, regions, nm_groups);
	  t = cvGetTickCount() - t;
	  printf( "GroupingNM done in %g ms.\n", t/((double)cvGetTickFrequency()*1000.) );

    cout << "Done!" << endl << endl;
    cout << "Press 'e' to show the extracted Extremal Regions, any other key to exit." << endl << endl;
    if( waitKey (-1) == 101)
        er_show(channels,regions);

    // memory clean-up
    er_filter1.release();
    er_filter2.release();
    regions.clear();
    if (!groups.empty())
    {
        groups.clear();
    }
}



// helper functions

void show_help_and_exit(const char *cmd)
{
    cout << "    Usage: " << cmd << " <input_image> " << endl;
    cout << "    Default classifier files (trained_classifierNM*.xml) must be in current directory" << endl << endl;
    exit(-1);
}

void groups_draw(Mat &src, vector<Rect> &groups)
{
    for (int i=groups.size()-1; i>=0; i--)
    {
        if (src.type() == CV_8UC3)
            rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 0, 255, 255 ), 3, 8 );
        else
            rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 255 ), 3, 8 );
    }
}

void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions)
{
    for (int c=0; c<(int)channels.size(); c++)
    {
        Mat dst = Mat::zeros(channels[0].rows+2,channels[0].cols+2,CV_8UC1);
        for (int r=0; r<(int)regions[c].size(); r++)
        {
            ERStat er = regions[c][r];
            if (er.parent != NULL) // deprecate the root region
            {
                int newMaskVal = 255;
                int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
                floodFill(channels[c],dst,Point(er.pixel%channels[c].cols,er.pixel/channels[c].cols),
                          Scalar(255),0,Scalar(er.level),Scalar(0),flags);
            }
        }
        char buff[10]; char *buff_ptr = buff;
        sprintf(buff, "channel %d", c);
        imshow(buff_ptr, dst);
    }
    waitKey(-1);
}
