/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <fstream>

#if defined _MSC_VER && _MSC_VER == 1500
    typedef int int_fast32_t;
#else
    #ifndef INT32_MAX
    #define __STDC_LIMIT_MACROS
    #include <stdint.h>
    #endif
#endif

using namespace std;

namespace cv
{

ERStat::ERStat(int init_level, int init_pixel, int init_x, int init_y) : pixel(init_pixel),
               level(init_level), area(0), perimeter(0), euler(0), probability(1.0),
               parent(0), child(0), next(0), prev(0), local_maxima(0),
               max_probability_ancestor(0), min_probability_ancestor(0)
{
    rect = Rect(init_x,init_y,1,1);
    raw_moments[0] = 0.0;
    raw_moments[1] = 0.0;
    central_moments[0] = 0.0;
    central_moments[1] = 0.0;
    central_moments[2] = 0.0;
    crossings = new std::deque<int>();
    crossings->push_back(0);
}


// derivative classes


// the classe implementing the interface for the 1st and 2nd stages of Neumann and Matas algorithm
class CV_EXPORTS ERFilterNM : public ERFilter
{
public:
    //Constructor
    ERFilterNM();
    //Destructor
    ~ERFilterNM() {}

    float minProbability;
    bool  nonMaxSuppression;
    float minProbabilityDiff;

    // the key method. Takes image on input, vector of ERStat is output for the first stage,
    // input/output - for the second one.
    void run( InputArray image, std::vector<ERStat>& regions );

protected:
    int thresholdDelta;
    float maxArea;
    float minArea;

    Ptr<ERFilter::Callback> classifier;

    // count of the rejected/accepted regions
    int num_rejected_regions;
    int num_accepted_regions;

public:

    // set/get methods to set the algorithm properties,
    void setCallback(const Ptr<ERFilter::Callback>& cb);
    void setThresholdDelta(int thresholdDelta);
    void setMinArea(float minArea);
    void setMaxArea(float maxArea);
    void setMinProbability(float minProbability);
    void setMinProbabilityDiff(float minProbabilityDiff);
    void setNonMaxSuppression(bool nonMaxSuppression);
    int  getNumRejected();

private:
    // pointer to the input/output regions vector
    std::vector<ERStat> *regions;
    // image mask used for feature calculations
    Mat region_mask;

    // extract the component tree and store all the ER regions
    void er_tree_extract( InputArray image );
    // accumulate a pixel into an ER
    void er_add_pixel( ERStat *parent, int x, int y, int non_boundary_neighbours,
                       int non_boundary_neighbours_horiz,
                       int d_C1, int d_C2, int d_C3 );
    // merge an ER with its nested parent
    void er_merge( ERStat *parent, ERStat *child );
    // copy extracted regions into the output vector
    ERStat* er_save( ERStat *er, ERStat *parent, ERStat *prev );
    // recursively walk the tree and filter (remove) regions using the callback classifier
    ERStat* er_tree_filter( InputArray image, ERStat *stat, ERStat *parent, ERStat *prev );
    // recursively walk the tree selecting only regions with local maxima probability
    ERStat* er_tree_nonmax_suppression( ERStat *er, ERStat *parent, ERStat *prev );
};


// default 1st stage classifier
class CV_EXPORTS ERClassifierNM1 : public ERFilter::Callback
{
public:
    //Constructor
    ERClassifierNM1(const std::string& filename);
    // Destructor
    ~ERClassifierNM1() {}

    // The classifier must return probability measure for the region.
    double eval(const ERStat& stat);

private:
    CvBoost boost;
};

// default 2nd stage classifier
class CV_EXPORTS ERClassifierNM2 : public ERFilter::Callback
{
public:
    //constructor
    ERClassifierNM2(const std::string& filename);
    // Destructor
    ~ERClassifierNM2() {}

    // The classifier must return probability measure for the region.
    double eval(const ERStat& stat);

private:
    CvBoost boost;
};





// default constructor
ERFilterNM::ERFilterNM()
{
    thresholdDelta = 1;
    minArea = 0.;
    maxArea = 1.;
    minProbability = 0.;
    nonMaxSuppression = false;
    minProbabilityDiff = 1.;
    num_accepted_regions = 0;
    num_rejected_regions = 0;
}

// the key method. Takes image on input, vector of ERStat is output for the first stage,
// input/output for the second one.
void ERFilterNM::run( InputArray image, std::vector<ERStat>& _regions )
{

    // assert correct image type
    CV_Assert( image.getMat().type() == CV_8UC1 );

    regions = &_regions;
    region_mask = Mat::zeros(image.getMat().rows+2, image.getMat().cols+2, CV_8UC1);

    // if regions vector is empty we must extract the entire component tree
    if ( regions->size() == 0 )
    {
        er_tree_extract( image );
        if (nonMaxSuppression)
        {
            vector<ERStat> aux_regions;
            regions->swap(aux_regions);
            regions->reserve(aux_regions.size());
            er_tree_nonmax_suppression( &aux_regions.front(), NULL, NULL );
            aux_regions.clear();
        }
    }
    else // if regions vector is already filled we'll just filter the current regions
    {
        // the tree root must have no parent
        CV_Assert( regions->front().parent == NULL );

        vector<ERStat> aux_regions;
        regions->swap(aux_regions);
        regions->reserve(aux_regions.size());
        er_tree_filter( image, &aux_regions.front(), NULL, NULL );
        aux_regions.clear();
    }
}

// extract the component tree and store all the ER regions
// uses the algorithm described in
// Linear time maximally stable extremal regions, D Nistér, H Stewénius – ECCV 2008
void ERFilterNM::er_tree_extract( InputArray image )
{

    Mat src = image.getMat();
    // assert correct image type
    CV_Assert( src.type() == CV_8UC1 );

    if (thresholdDelta > 1)
    {
        src = (src / thresholdDelta) -1;
    }

    const unsigned char * image_data = src.data;
    int width = src.cols, height = src.rows;

    // the component stack
    vector<ERStat*> er_stack;

    //the quads for euler number calculation
    unsigned char quads[3][4];
    quads[0][0] = 1 << 3;
    quads[0][1] = 1 << 2;
    quads[0][2] = 1 << 1;
    quads[0][3] = 1;
    quads[1][0] = (1<<2)|(1<<1)|(1);
    quads[1][1] = (1<<3)|(1<<1)|(1);
    quads[1][2] = (1<<3)|(1<<2)|(1);
    quads[1][3] = (1<<3)|(1<<2)|(1<<1);
    quads[2][0] = (1<<2)|(1<<1);
    quads[2][1] = (1<<3)|(1);
    quads[2][3] = 255;


    // masks to know if a pixel is accessible and if it has been already added to some region
    vector<bool> accessible_pixel_mask(width * height);
    vector<bool> accumulated_pixel_mask(width * height);

    // heap of boundary pixels
    vector<int> boundary_pixes[256];
    vector<int> boundary_edges[256];

    // add a dummy-component before start
    er_stack.push_back(new ERStat);

    // we'll look initially for all pixels with grey-level lower than a grey-level higher than any allowed in the image
    int threshold_level = (255/thresholdDelta)+1;

    // starting from the first pixel (0,0)
    int current_pixel = 0;
    int current_edge = 0;
    int current_level = image_data[0];
    accessible_pixel_mask[0] = true;

    bool push_new_component = true;

    for (;;) {

        int x = current_pixel % width;
        int y = current_pixel / width;

        // push a component with current level in the component stack
        if (push_new_component)
            er_stack.push_back(new ERStat(current_level, current_pixel, x, y));
        push_new_component = false;

        // explore the (remaining) edges to the neighbors to the current pixel
        for ( ; current_edge < 4; current_edge++)
        {

            int neighbour_pixel = current_pixel;

            switch (current_edge)
            {
                    case 0: if (x < width - 1) neighbour_pixel = current_pixel + 1;  break;
                    case 1: if (y < height - 1) neighbour_pixel = current_pixel + width; break;
                    case 2: if (x > 0) neighbour_pixel = current_pixel - 1; break;
                    default: if (y > 0) neighbour_pixel = current_pixel - width; break;
            }

            // if neighbour is not accessible, mark it accessible and retreive its grey-level value
            if ( !accessible_pixel_mask[neighbour_pixel] && (neighbour_pixel != current_pixel) )
            {

                int neighbour_level = image_data[neighbour_pixel];
                accessible_pixel_mask[neighbour_pixel] = true;

                // if neighbour level is not lower than current level add neighbour to the boundary heap
                if (neighbour_level >= current_level)
                {

                    boundary_pixes[neighbour_level].push_back(neighbour_pixel);
                    boundary_edges[neighbour_level].push_back(0);

                    // if neighbour level is lower than our threshold_level set threshold_level to neighbour level
                    if (neighbour_level < threshold_level)
                        threshold_level = neighbour_level;

                }
                else // if neighbour level is lower than current add current_pixel (and next edge)
                     // to the boundary heap for later processing
                {

                    boundary_pixes[current_level].push_back(current_pixel);
                    boundary_edges[current_level].push_back(current_edge + 1);

                    // if neighbour level is lower than threshold_level set threshold_level to neighbour level
                    if (current_level < threshold_level)
                        threshold_level = current_level;

                    // consider the new pixel and its grey-level as current pixel
                    current_pixel = neighbour_pixel;
                    current_edge = 0;
                    current_level = neighbour_level;

                    // and push a new component
                    push_new_component = true;
                    break;
                }
            }

        } // else neigbor was already accessible

        if (push_new_component) continue;


        // once here we can add the current pixel to the component at the top of the stack
        // but first we find how many of its neighbours are part of the region boundary (needed for
        // perimeter and crossings calc.) and the increment in quads counts for euler number calc.
        int non_boundary_neighbours = 0;
        int non_boundary_neighbours_horiz = 0;

        unsigned char quad_before[4] = {0,0,0,0};
        unsigned char quad_after[4] = {0,0,0,0};
        quad_after[0] = 1<<1;
        quad_after[1] = 1<<3;
        quad_after[2] = 1<<2;
        quad_after[3] = 1;

        for (int edge = 0; edge < 8; edge++)
        {
            int neighbour4 = -1;
            int neighbour8 = -1;
            int cell = 0;
            switch (edge)
            {
                    case 0: if (x < width - 1) { neighbour4 = neighbour8 = current_pixel + 1;} cell = 5; break;
                    case 1: if ((x < width - 1)&&(y < height - 1)) { neighbour8 = current_pixel + 1 + width;} cell = 8; break;
                    case 2: if (y < height - 1) { neighbour4 = neighbour8 = current_pixel + width;} cell = 7; break;
                    case 3: if ((x > 0)&&(y < height - 1)) { neighbour8 = current_pixel - 1 + width;} cell = 6; break;
                    case 4: if (x > 0) { neighbour4 = neighbour8 = current_pixel - 1;} cell = 3; break;
                    case 5: if ((x > 0)&&(y > 0)) { neighbour8 = current_pixel - 1 - width;} cell = 0; break;
                    case 6: if (y > 0) { neighbour4 = neighbour8 = current_pixel - width;} cell = 1; break;
                    default: if ((x < width - 1)&&(y > 0)) { neighbour8 = current_pixel + 1 - width;} cell = 2; break;
            }
            if ((neighbour4 != -1)&&(accumulated_pixel_mask[neighbour4])&&(image_data[neighbour4]<=image_data[current_pixel]))
            {
                non_boundary_neighbours++;
                if ((edge == 0) || (edge == 4))
                    non_boundary_neighbours_horiz++;
            }

            int pix_value = image_data[current_pixel] + 1;
            if (neighbour8 != -1)
            {
                if (accumulated_pixel_mask[neighbour8])
                    pix_value = image_data[neighbour8];
            }

            if (pix_value<=image_data[current_pixel])
            {
                switch(cell)
                {
                    case 0:
                        quad_before[3] = quad_before[3] | (1<<3);
                        quad_after[3]  = quad_after[3]  | (1<<3);
                        break;
                    case 1:
                        quad_before[3] = quad_before[3] | (1<<2);
                        quad_after[3]  = quad_after[3]  | (1<<2);
                        quad_before[0] = quad_before[0] | (1<<3);
                        quad_after[0]  = quad_after[0]  | (1<<3);
                        break;
                    case 2:
                        quad_before[0] = quad_before[0] | (1<<2);
                        quad_after[0]  = quad_after[0]  | (1<<2);
                        break;
                    case 3:
                        quad_before[3] = quad_before[3] | (1<<1);
                        quad_after[3]  = quad_after[3]  | (1<<1);
                        quad_before[2] = quad_before[2] | (1<<3);
                        quad_after[2]  = quad_after[2]  | (1<<3);
                        break;
                    case 5:
                        quad_before[0] = quad_before[0] | (1);
                        quad_after[0]  = quad_after[0]  | (1);
                        quad_before[1] = quad_before[1] | (1<<2);
                        quad_after[1]  = quad_after[1]  | (1<<2);
                        break;
                    case 6:
                        quad_before[2] = quad_before[2] | (1<<1);
                        quad_after[2]  = quad_after[2]  | (1<<1);
                        break;
                    case 7:
                        quad_before[2] = quad_before[2] | (1);
                        quad_after[2]  = quad_after[2]  | (1);
                        quad_before[1] = quad_before[1] | (1<<1);
                        quad_after[1]  = quad_after[1]  | (1<<1);
                        break;
                    default:
                        quad_before[1] = quad_before[1] | (1);
                        quad_after[1]  = quad_after[1]  | (1);
                        break;
                }
            }

        }

        int C_before[3] = {0, 0, 0};
        int C_after[3] = {0, 0, 0};

        for (int p=0; p<3; p++)
        {
            for (int q=0; q<4; q++)
            {
                if ( (quad_before[0] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[1] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[2] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[3] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;

                if ( (quad_after[0] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if ( (quad_after[1] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if ( (quad_after[2] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if ( (quad_after[3] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
            }
        }

        int d_C1 = C_after[0]-C_before[0];
        int d_C2 = C_after[1]-C_before[1];
        int d_C3 = C_after[2]-C_before[2];

        er_add_pixel(er_stack.back(), x, y, non_boundary_neighbours, non_boundary_neighbours_horiz, d_C1, d_C2, d_C3);
        accumulated_pixel_mask[current_pixel] = true;

        // if we have processed all the possible threshold levels (the hea is empty) we are done!
        if (threshold_level == (255/thresholdDelta)+1)
        {

            // save the extracted regions into the output vector
            regions->reserve(num_accepted_regions+1);
            er_save(er_stack.back(), NULL, NULL);

            // clean memory
            for (size_t r=0; r<er_stack.size(); r++)
            {
                ERStat *stat = er_stack.at(r);
                if (stat->crossings)
                {
                    stat->crossings->clear();
                    delete(stat->crossings);
                    stat->crossings = NULL;
                }
                delete stat;
            }
            er_stack.clear();

            return;
        }


        // pop the heap of boundary pixels
        current_pixel = boundary_pixes[threshold_level].back();
        boundary_pixes[threshold_level].erase(boundary_pixes[threshold_level].end()-1);
        current_edge  = boundary_edges[threshold_level].back();
        boundary_edges[threshold_level].erase(boundary_edges[threshold_level].end()-1);

        while (boundary_pixes[threshold_level].empty() && (threshold_level < (255/thresholdDelta)+1))
            threshold_level++;


        int new_level = image_data[current_pixel];

        // if the new pixel has higher grey value than the current one
        if (new_level != current_level) {

            current_level = new_level;

            // process components on the top of the stack until we reach the higher grey-level
            while (er_stack.back()->level < new_level)
            {
                ERStat* er = er_stack.back();
                er_stack.erase(er_stack.end()-1);

                if (new_level < er_stack.back()->level)
                {
                    er_stack.push_back(new ERStat(new_level, current_pixel, current_pixel%width, current_pixel/width));
                    er_merge(er_stack.back(), er);
                    break;
                }

                er_merge(er_stack.back(), er);
            }

        }

    }
}

// accumulate a pixel into an ER
void ERFilterNM::er_add_pixel(ERStat *parent, int x, int y, int non_border_neighbours,
                                                            int non_border_neighbours_horiz,
                                                            int d_C1, int d_C2, int d_C3)
{
    parent->area++;
    parent->perimeter += 4 - 2*non_border_neighbours;

    if (parent->crossings->size()>0)
    {
        if (y<parent->rect.y) parent->crossings->push_front(2);
        else if (y>parent->rect.br().y-1) parent->crossings->push_back(2);
        else {
            parent->crossings->at(y - parent->rect.y) += 2-2*non_border_neighbours_horiz;
        }
    } else {
        parent->crossings->push_back(2);
    }

    parent->euler += (d_C1 - d_C2 + 2*d_C3) / 4;

    int new_x1 = min(parent->rect.x,x);
    int new_y1 = min(parent->rect.y,y);
    int new_x2 = max(parent->rect.br().x-1,x);
    int new_y2 = max(parent->rect.br().y-1,y);
    parent->rect.x = new_x1;
    parent->rect.y = new_y1;
    parent->rect.width  = new_x2-new_x1+1;
    parent->rect.height = new_y2-new_y1+1;

    parent->raw_moments[0] += x;
    parent->raw_moments[1] += y;

    parent->central_moments[0] += x * x;
    parent->central_moments[1] += x * y;
    parent->central_moments[2] += y * y;
}

// merge an ER with its nested parent
void ERFilterNM::er_merge(ERStat *parent, ERStat *child)
{

    parent->area += child->area;

    parent->perimeter += child->perimeter;


    for (int i=parent->rect.y; i<=min(parent->rect.br().y-1,child->rect.br().y-1); i++)
        if (i-child->rect.y >= 0)
            parent->crossings->at(i-parent->rect.y) += child->crossings->at(i-child->rect.y);

    for (int i=parent->rect.y-1; i>=child->rect.y; i--)
        if (i-child->rect.y < (int)child->crossings->size())
            parent->crossings->push_front(child->crossings->at(i-child->rect.y));
        else
            parent->crossings->push_front(0);

    for (int i=parent->rect.br().y; i<child->rect.y; i++)
        parent->crossings->push_back(0);

    for (int i=max(parent->rect.br().y,child->rect.y); i<=child->rect.br().y-1; i++)
        parent->crossings->push_back(child->crossings->at(i-child->rect.y));

    parent->euler += child->euler;

    int new_x1 = min(parent->rect.x,child->rect.x);
    int new_y1 = min(parent->rect.y,child->rect.y);
    int new_x2 = max(parent->rect.br().x-1,child->rect.br().x-1);
    int new_y2 = max(parent->rect.br().y-1,child->rect.br().y-1);
    parent->rect.x = new_x1;
    parent->rect.y = new_y1;
    parent->rect.width  = new_x2-new_x1+1;
    parent->rect.height = new_y2-new_y1+1;

    parent->raw_moments[0] += child->raw_moments[0];
    parent->raw_moments[1] += child->raw_moments[1];

    parent->central_moments[0] += child->central_moments[0];
    parent->central_moments[1] += child->central_moments[1];
    parent->central_moments[2] += child->central_moments[2];

    vector<int> m_crossings;
    m_crossings.push_back(child->crossings->at((int)(child->rect.height)/6));
    m_crossings.push_back(child->crossings->at((int)3*(child->rect.height)/6));
    m_crossings.push_back(child->crossings->at((int)5*(child->rect.height)/6));
    std::sort(m_crossings.begin(), m_crossings.end());
    child->med_crossings = (float)m_crossings.at(1);

    // free unnecessary mem
    child->crossings->clear();
    delete(child->crossings);
    child->crossings = NULL;

    // recover the original grey-level
    child->level = child->level*thresholdDelta;

    // before saving calculate P(child|character) and filter if possible
    if (classifier != NULL)
    {
        child->probability = classifier->eval(*child);
    }

    if ( (((classifier!=NULL)?(child->probability >= minProbability):true)||(nonMaxSuppression)) &&
         ((child->area >= (minArea*region_mask.rows*region_mask.cols)) &&
          (child->area <= (maxArea*region_mask.rows*region_mask.cols)) &&
          (child->rect.width > 2) && (child->rect.height > 2)) )
    {

        num_accepted_regions++;

        child->next = parent->child;
        if (parent->child)
            parent->child->prev = child;
        parent->child = child;
        child->parent = parent;

    } else {

        num_rejected_regions++;

        if (child->prev !=NULL)
            child->prev->next = child->next;

        ERStat *new_child = child->child;
        if (new_child != NULL)
        {
            while (new_child->next != NULL)
                new_child = new_child->next;
            new_child->next = parent->child;
            if (parent->child)
                parent->child->prev = new_child;
            parent->child   = child->child;
            child->child->parent = parent;
        }

        // free mem
        if(child->crossings)
        {
            child->crossings->clear();
            delete(child->crossings);
            child->crossings = NULL;
        }
        delete(child);
    }

}

// copy extracted regions into the output vector
ERStat* ERFilterNM::er_save( ERStat *er, ERStat *parent, ERStat *prev )
{

    regions->push_back(*er);

    regions->back().parent = parent;
    if (prev != NULL)
    {
      prev->next = &(regions->back());
    }
    else if (parent != NULL)
      parent->child = &(regions->back());

    ERStat *old_prev = NULL;
    ERStat *this_er  = &regions->back();

    if (this_er->parent == NULL)
    {
       this_er->probability = 0;
    }

    if (nonMaxSuppression)
    {
        if (this_er->parent == NULL)
        {
            this_er->max_probability_ancestor = this_er;
            this_er->min_probability_ancestor = this_er;
        }
        else
        {
            this_er->max_probability_ancestor = (this_er->probability > parent->max_probability_ancestor->probability)? this_er :  parent->max_probability_ancestor;

            this_er->min_probability_ancestor = (this_er->probability < parent->min_probability_ancestor->probability)? this_er :  parent->min_probability_ancestor;

            if ( (this_er->max_probability_ancestor->probability > minProbability) && (this_er->max_probability_ancestor->probability - this_er->min_probability_ancestor->probability > minProbabilityDiff))
            {
              this_er->max_probability_ancestor->local_maxima = true;
              if ((this_er->max_probability_ancestor == this_er) && (this_er->parent->local_maxima))
              {
                this_er->parent->local_maxima = false;
              }
            }
            else if (this_er->probability < this_er->parent->probability)
            {
              this_er->min_probability_ancestor = this_er;
            }
            else if (this_er->probability > this_er->parent->probability)
            {
              this_er->max_probability_ancestor = this_er;
            }


        }
    }

    for (ERStat * child = er->child; child; child = child->next)
    {
        old_prev = er_save(child, this_er, old_prev);
    }

    return this_er;
}

// recursively walk the tree and filter (remove) regions using the callback classifier
ERStat* ERFilterNM::er_tree_filter ( InputArray image, ERStat * stat, ERStat *parent, ERStat *prev )
{
    Mat src = image.getMat();
    // assert correct image type
    CV_Assert( src.type() == CV_8UC1 );

    //Fill the region and calculate 2nd stage features
    Mat region = region_mask(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x+2,stat->rect.br().y+2)));
    region = Scalar(0);
    int newMaskVal = 255;
    int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
    Rect rect;

    floodFill( src(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x,stat->rect.br().y))),
               region, Point(stat->pixel%src.cols - stat->rect.x, stat->pixel/src.cols - stat->rect.y),
               Scalar(255), &rect, Scalar(stat->level), Scalar(0), flags );
    rect.width += 2;
    rect.height += 2;
    region = region(rect);

    vector<vector<Point> > contours;
    vector<Point> contour_poly;
    vector<Vec4i> hierarchy;
    findContours( region, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0) );
    //TODO check epsilon parameter of approxPolyDP (set empirically) : we want more precission
    //     if the region is very small because otherwise we'll loose all the convexities
    approxPolyDP( Mat(contours[0]), contour_poly, (float)min(rect.width,rect.height)/17, true );

    bool was_convex = false;
    int  num_inflexion_points = 0;

    for (int p = 0 ; p<(int)contour_poly.size(); p++)
    {
        int p_prev = p-1;
        int p_next = p+1;
        if (p_prev == -1)
            p_prev = (int)contour_poly.size()-1;
        if (p_next == (int)contour_poly.size())
            p_next = 0;

        double angle_next = atan2((double)(contour_poly[p_next].y-contour_poly[p].y),
                                  (double)(contour_poly[p_next].x-contour_poly[p].x));
        double angle_prev = atan2((double)(contour_poly[p_prev].y-contour_poly[p].y),
                                  (double)(contour_poly[p_prev].x-contour_poly[p].x));
        if ( angle_next < 0 )
            angle_next = 2.*CV_PI + angle_next;

        double angle = (angle_next - angle_prev);
        if (angle > 2.*CV_PI)
            angle = angle - 2.*CV_PI;
        else if (angle < 0)
            angle = 2.*CV_PI + std::abs(angle);

        if (p>0)
        {
            if ( ((angle > CV_PI)&&(!was_convex)) || ((angle < CV_PI)&&(was_convex)) )
                num_inflexion_points++;
        }
        was_convex = (angle > CV_PI);

    }

    floodFill(region, Point(0,0), Scalar(255), 0);
    int holes_area = region.cols*region.rows-countNonZero(region);

    int hull_area = 0;

    {

        vector<Point> hull;
        convexHull(contours[0], hull, false);
        hull_area = (int)contourArea(hull);
    }


    stat->hole_area_ratio = (float)holes_area / stat->area;
    stat->convex_hull_ratio = (float)hull_area / (float)contourArea(contours[0]);
    stat->num_inflexion_points = (float)num_inflexion_points;


    // calculate P(child|character) and filter if possible
    if ( (classifier != NULL) && (stat->parent != NULL) )
    {
        stat->probability = classifier->eval(*stat);
    }

    if ( ( ((classifier != NULL)?(stat->probability >= minProbability):true) &&
          ((stat->area >= minArea*region_mask.rows*region_mask.cols) &&
           (stat->area <= maxArea*region_mask.rows*region_mask.cols)) ) ||
        (stat->parent == NULL) )
    {

        num_accepted_regions++;
        regions->push_back(*stat);

        regions->back().parent = parent;
        regions->back().next   = NULL;
        regions->back().child  = NULL;

        if (prev != NULL)
            prev->next = &(regions->back());
        else if (parent != NULL)
            parent->child = &(regions->back());

        ERStat *old_prev = NULL;
        ERStat *this_er  = &regions->back();

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_filter(image, child, this_er, old_prev);
        }

        return this_er;

    } else {

        num_rejected_regions++;

        ERStat *old_prev = prev;

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_filter(image, child, parent, old_prev);
        }

        return old_prev;
    }

}

// recursively walk the tree selecting only regions with local maxima probability
ERStat* ERFilterNM::er_tree_nonmax_suppression ( ERStat * stat, ERStat *parent, ERStat *prev )
{

    if ( ( stat->local_maxima ) || ( stat->parent == NULL ) )
    {

        regions->push_back(*stat);

        regions->back().parent = parent;
        regions->back().next   = NULL;
        regions->back().child  = NULL;

        if (prev != NULL)
            prev->next = &(regions->back());
        else if (parent != NULL)
            parent->child = &(regions->back());

        ERStat *old_prev = NULL;
        ERStat *this_er  = &regions->back();

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression( child, this_er, old_prev );
        }

        return this_er;

    } else {

        num_rejected_regions++;
        num_accepted_regions--;

        ERStat *old_prev = prev;

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression( child, parent, old_prev );
        }

        return old_prev;
    }

}

void ERFilterNM::setCallback(const Ptr<ERFilter::Callback>& cb)
{
    classifier = cb;
}

void ERFilterNM::setMinArea(float _minArea)
{
    CV_Assert( (_minArea >= 0) && (_minArea < maxArea) );
    minArea = _minArea;
    return;
}

void ERFilterNM::setMaxArea(float _maxArea)
{
    CV_Assert(_maxArea <= 1);
    CV_Assert(minArea < _maxArea);
    maxArea = _maxArea;
    return;
}

void ERFilterNM::setThresholdDelta(int _thresholdDelta)
{
    CV_Assert( (_thresholdDelta > 0) && (_thresholdDelta <= 128) );
    thresholdDelta = _thresholdDelta;
    return;
}

void ERFilterNM::setMinProbability(float _minProbability)
{
    CV_Assert( (_minProbability >= 0.0) && (_minProbability <= 1.0) );
    minProbability = _minProbability;
    return;
}

void ERFilterNM::setMinProbabilityDiff(float _minProbabilityDiff)
{
    CV_Assert( (_minProbabilityDiff >= 0.0) && (_minProbabilityDiff <= 1.0) );
    minProbabilityDiff = _minProbabilityDiff;
    return;
}

void ERFilterNM::setNonMaxSuppression(bool _nonMaxSuppression)
{
    nonMaxSuppression = _nonMaxSuppression;
    return;
}

int ERFilterNM::getNumRejected()
{
    return num_rejected_regions;
}




// load default 1st stage classifier if found
ERClassifierNM1::ERClassifierNM1(const std::string& filename)
{

    if (ifstream(filename.c_str()))
        boost.load( filename.c_str(), "boost" );
    else
        CV_Error(CV_StsBadArg, "Default classifier file not found!");
}

double ERClassifierNM1::eval(const ERStat& stat)
{
    //Classify
    float arr[] = {0,(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings};

    vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );

    float votes = boost.predict( Mat(sample), Mat(), Range::all(), false, true );

    // Logistic Correction returns a probability value (in the range(0,1))
    return (double)1-(double)1/(1+exp(-2*votes));
}


// load default 2nd stage classifier if found
ERClassifierNM2::ERClassifierNM2(const std::string& filename)
{
    if (ifstream(filename.c_str()))
        boost.load( filename.c_str(), "boost" );
    else
        CV_Error(CV_StsBadArg, "Default classifier file not found!");
}

double ERClassifierNM2::eval(const ERStat& stat)
{
    //Classify
    float arr[] = {0,(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings, stat.hole_area_ratio,
                     stat.convex_hull_ratio, stat.num_inflexion_points};

    vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );

    float votes = boost.predict( Mat(sample), Mat(), Range::all(), false, true );

    // Logistic Correction returns a probability value (in the range(0,1))
    return (double)1-(double)1/(1+exp(-2*votes));
}


/*!
    Create an Extremal Region Filter for the 1st stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    The component tree of the image is extracted by a threshold increased step by step
    from 0 to 255, incrementally computable descriptors (aspect_ratio, compactness,
    number of holes, and number of horizontal crossings) are computed for each ER
    and used as features for a classifier which estimates the class-conditional
    probability P(er|character). The value of P(er|character) is tracked using the inclusion
    relation of ER across all thresholds and only the ERs which correspond to local maximum
    of the probability P(er|character) are selected (if the local maximum of the
    probability is above a global limit pmin and the difference between local maximum and
    local minimum is greater than minProbabilityDiff).

    \param  cb                Callback with the classifier.
                              default classifier can be implicitly load with function loadClassifierNM1()
                              from file in samples/cpp/trained_classifierNM1.xml
    \param  thresholdDelta    Threshold step in subsequent thresholds when extracting the component tree
    \param  minArea           The minimum area (% of image size) allowed for retreived ER's
    \param  minArea           The maximum area (% of image size) allowed for retreived ER's
    \param  minProbability    The minimum probability P(er|character) allowed for retreived ER's
    \param  nonMaxSuppression Whenever non-maximum suppression is done over the branch probabilities
    \param  minProbability    The minimum probability difference between local maxima and local minima ERs
*/
Ptr<ERFilter> createERFilterNM1(const Ptr<ERFilter::Callback>& cb, int thresholdDelta,
                                float minArea, float maxArea, float minProbability,
                                bool nonMaxSuppression, float minProbabilityDiff)
{

    CV_Assert( (minProbability >= 0.) && (minProbability <= 1.) );
    CV_Assert( (minArea < maxArea) && (minArea >=0.) && (maxArea <= 1.) );
    CV_Assert( (thresholdDelta >= 0) && (thresholdDelta <= 128) );
    CV_Assert( (minProbabilityDiff >= 0.) && (minProbabilityDiff <= 1.) );

    Ptr<ERFilterNM> filter = makePtr<ERFilterNM>();

    filter->setCallback(cb);

    filter->setThresholdDelta(thresholdDelta);
    filter->setMinArea(minArea);
    filter->setMaxArea(maxArea);
    filter->setMinProbability(minProbability);
    filter->setNonMaxSuppression(nonMaxSuppression);
    filter->setMinProbabilityDiff(minProbabilityDiff);
    return (Ptr<ERFilter>)filter;
}

/*!
    Create an Extremal Region Filter for the 2nd stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    In the second stage, the ERs that passed the first stage are classified into character
    and non-character classes using more informative but also more computationally expensive
    features. The classifier uses all the features calculated in the first stage and the following
    additional features: hole area ratio, convex hull ratio, and number of outer inflexion points.

    \param  cb             Callback with the classifier
                           default classifier can be implicitly load with function loadClassifierNM1()
                           from file in samples/cpp/trained_classifierNM2.xml
    \param  minProbability The minimum probability P(er|character) allowed for retreived ER's
*/
Ptr<ERFilter> createERFilterNM2(const Ptr<ERFilter::Callback>& cb, float minProbability)
{

    CV_Assert( (minProbability >= 0.) && (minProbability <= 1.) );

    Ptr<ERFilterNM> filter = makePtr<ERFilterNM>();

    filter->setCallback(cb);

    filter->setMinProbability(minProbability);
    return (Ptr<ERFilter>)filter;
}

/*!
    Allow to implicitly load the default classifier when creating an ERFilter object.
    The function takes as parameter the XML or YAML file with the classifier model
    (e.g. trained_classifierNM1.xml) returns a pointer to ERFilter::Callback.
*/
Ptr<ERFilter::Callback> loadClassifierNM1(const std::string& filename)

{
    return makePtr<ERClassifierNM1>(filename);
}

/*!
    Allow to implicitly load the default classifier when creating an ERFilter object.
    The function takes as parameter the XML or YAML file with the classifier model
    (e.g. trained_classifierNM2.xml) returns a pointer to ERFilter::Callback.
*/
Ptr<ERFilter::Callback> loadClassifierNM2(const std::string& filename)
{
    return makePtr<ERClassifierNM2>(filename);
}


/* ------------------------------------------------------------------------------------*/
/* -------------------------------- Compute Channels NM -------------------------------*/
/* ------------------------------------------------------------------------------------*/


void  get_gradient_magnitude(Mat& _grey_img, Mat& _gradient_magnitude);

void get_gradient_magnitude(Mat& _grey_img, Mat& _gradient_magnitude)
{
    Mat C = Mat_<float>(_grey_img);

    Mat kernel = (Mat_<float>(1,3) << -1,0,1);
    Mat grad_x;
    filter2D(C, grad_x, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT);

    Mat kernel2 = (Mat_<float>(3,1) << -1,0,1);
    Mat grad_y;
    filter2D(C, grad_y, -1, kernel2, Point(-1,-1), 0, BORDER_DEFAULT);

    magnitude( grad_x, grad_y, _gradient_magnitude);
}


/*!
    Compute the diferent channels to be processed independently in the N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    In N&M algorithm, the combination of intensity (I), hue (H), saturation (S), and gradient
    magnitude channels (Grad) are used in order to obatin high localization recall.
    This implementation also the alternative combination of red (R), grren (G), blue (B),
    lightness (L), and gradient magnitude (Grad).

    \param  _src           Source image. Must be RGB CV_8UC3.
    \param  _channels      Output vector<Mat> where computed channels are stored.
    \param  _mode          Mode of operation. Currently the only available options are
                           ERFILTER_NM_RGBLGrad and ERFILTER_NM_IHSGrad.

*/
void computeNMChannels(InputArray _src, OutputArrayOfArrays _channels, int _mode)
{

    CV_Assert( ( _mode == ERFILTER_NM_RGBLGrad ) || ( _mode == ERFILTER_NM_IHSGrad ) );

    Mat src = _src.getMat();
    if( src.empty() )
    {
        _channels.release();
        return;
    }

    // assert RGB image
    CV_Assert(src.type() == CV_8UC3);

    if (_mode == ERFILTER_NM_IHSGrad)
    {
        _channels.create( 4, 1, src.depth());

        Mat hsv;
        cvtColor(src, hsv, COLOR_RGB2HSV);
        vector<Mat> channelsHSV;
        split(hsv, channelsHSV);

        for (int i = 0; i < src.channels(); i++)
        {
            _channels.create(src.rows, src.cols, CV_8UC1, i);
            Mat channel = _channels.getMat(i);
            channelsHSV.at(i).copyTo(channel);
        }

        Mat grey;
        cvtColor(src, grey, COLOR_RGB2GRAY);
        Mat gradient_magnitude = Mat_<float>(grey.size());
        get_gradient_magnitude( grey, gradient_magnitude);
        gradient_magnitude.convertTo(gradient_magnitude, CV_8UC1);

        _channels.create(src.rows, src.cols, CV_8UC1, 3);
        Mat channelGrad = _channels.getMat(3);
        gradient_magnitude.copyTo(channelGrad);

    } else if (_mode == ERFILTER_NM_RGBLGrad) {

        _channels.create( 5, 1, src.depth());

        vector<Mat> channelsRGB;
        split(src, channelsRGB);
        for (int i = 0; i < src.channels(); i++)
        {
            _channels.create(src.rows, src.cols, CV_8UC1, i);
            Mat channel = _channels.getMat(i);
            channelsRGB.at(i).copyTo(channel);
        }

        Mat hls;
        cvtColor(src, hls, COLOR_RGB2HLS);
        vector<Mat> channelsHLS;
        split(hls, channelsHLS);

        _channels.create(src.rows, src.cols, CV_8UC1, 3);
        Mat channelL = _channels.getMat(3);
        channelsHLS.at(1).copyTo(channelL);

        Mat grey;
        cvtColor(src, grey, COLOR_RGB2GRAY);
        Mat gradient_magnitude = Mat_<float>(grey.size());
        get_gradient_magnitude( grey, gradient_magnitude);
        gradient_magnitude.convertTo(gradient_magnitude, CV_8UC1);

        _channels.create(src.rows, src.cols, CV_8UC1, 4);
        Mat channelGrad = _channels.getMat(4);
        gradient_magnitude.copyTo(channelGrad);
    }
}



/* ------------------------------------------------------------------------------------*/
/* -------------------------------- ER Grouping Algorithm -----------------------------*/
/* ------------------------------------------------------------------------------------*/


/*  NFA approximation functions */

// ln(10)
#ifndef M_LN10
#define M_LN10     2.30258509299404568401799145468436421
#endif
// Doubles relative error factor
#define RELATIVE_ERROR_FACTOR 100.0

// Compare doubles by relative error.
static int double_equal(double a, double b)
{
    double abs_diff,aa,bb,abs_max;

    /* trivial case */
    if( a == b ) return true;

    abs_diff = fabs(a-b);
    aa = fabs(a);
    bb = fabs(b);
    abs_max = aa > bb ? aa : bb;

    /* DBL_MIN is the smallest normalized number, thus, the smallest
       number whose relative error is bounded by DBL_EPSILON. For
       smaller numbers, the same quantization steps as for DBL_MIN
       are used. Then, for smaller numbers, a meaningful "relative"
       error should be computed by dividing the difference by DBL_MIN. */
    if( abs_max < DBL_MIN ) abs_max = DBL_MIN;

    /* equal if relative error <= factor x eps */
    return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}


/*
     Computes the natural logarithm of the absolute value of
     the gamma function of x using the Lanczos approximation.
     See http://www.rskey.org/gamma.htm
*/
static double log_gamma_lanczos(double x)
{
    static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                           8687.24529705, 1168.92649479, 83.8676043424,
                           2.50662827511 };
    double a = (x+0.5) * log(x+5.5) - (x+5.5);
    double b = 0.0;
    int n;

    for(n=0;n<7;n++)
    {
        a -= log( x + (double) n );
        b += q[n] * pow( x, (double) n );
    }
    return a + log(b);
}

/*
     Computes the natural logarithm of the absolute value of
     the gamma function of x using Windschitl method.
     See http://www.rskey.org/gamma.htm
*/
static double log_gamma_windschitl(double x)
{
    return 0.918938533204673 + (x-0.5)*log(x) - x
           + 0.5*x*log( x*sinh(1/x) + 1/(810.0*pow(x,6.0)) );
}

/*
     Computes the natural logarithm of the absolute value of
     the gamma function of x. When x>15 use log_gamma_windschitl(),
     otherwise use log_gamma_lanczos().
*/
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

// Size of the table to store already computed inverse values.
#define TABSIZE 100000

/*
     Computes -log10(NFA).
     NFA stands for Number of False Alarms:
*/
static double NFA(int n, int k, double p, double logNT)
{
    static double inv[TABSIZE];   /* table to keep computed inverse values */
    double tolerance = 0.1;       /* an error of 10% in the result is accepted */
    double log1term,term,bin_term,mult_term,bin_tail,err,p_term;
    int i;

    if (p<=0)
        p=0.000000000000000000000000000001;
    if (p>=1)
        p=0.999999999999999999999999999999;

    /* check parameters */
    if( n<0 || k<0 || k>n || p<=0.0 || p>=1.0 )
    {
        CV_Error(CV_StsBadArg, "erGrouping wrong n, k or p values in NFA call!");
    }

    /* trivial cases */
    if( n==0 || k==0 ) return -logNT;
    if( n==k ) return -logNT - (double) n * log10(p);

    /* probability term */
    p_term = p / (1.0-p);

    /* compute the first term of the series */
    log1term = log_gamma( (double) n + 1.0 ) - log_gamma( (double) k + 1.0 )
               - log_gamma( (double) (n-k) + 1.0 )
               + (double) k * log(p) + (double) (n-k) * log(1.0-p);
    term = exp(log1term);

    /* in some cases no more computations are needed */
    if( double_equal(term,0.0) )              /* the first term is almost zero */
    {
        if( (double) k > (double) n * p )     /* at begin or end of the tail?  */
            return -log1term / M_LN10 - logNT;  /* end: use just the first term  */
        else
            return -logNT;                      /* begin: the tail is roughly 1  */
    }

    /* compute more terms if needed */
    bin_tail = term;
    for(i=k+1;i<=n;i++)
    {
        bin_term = (double) (n-i+1) * ( i<TABSIZE ?
                    ( inv[i]!=0.0 ? inv[i] : ( inv[i] = 1.0 / (double) i ) ) :
                    1.0 / (double) i );

        mult_term = bin_term * p_term;
        term *= mult_term;
        bin_tail += term;
        if(bin_term<1.0)
        {
            err = term * ( ( 1.0 - pow( mult_term, (double) (n-i+1) ) ) /
                           (1.0-mult_term) - 1.0 );
            if( err < tolerance * fabs(-log10(bin_tail)-logNT) * bin_tail ) break;
        }
    }
    return -log10(bin_tail) - logNT;
}


// Minibox : smallest enclosing box of a set of n points in d dimensions

class Minibox {
private:
    vector<float> edge_begin;
    vector<float> edge_end;
    bool   initialized;

public:
    // creates an empty box
    Minibox();

    // copies p to the internal point set
    void check_in (vector<float> *p);

    // returns the volume of the box
    long double volume();
};

Minibox::Minibox()
{
    initialized = false;
}

void Minibox::check_in (vector<float> *p)
{
    if(!initialized) for (int i=0; i<(int)p->size(); i++)
    {
        edge_begin.push_back(p->at(i));
        edge_end.push_back(p->at(i)+0.00000000000000001f);
        initialized = true;
    }
    else for (int i=0; i<(int)p->size(); i++)
    {
        edge_begin.at(i) = min(p->at(i),edge_begin.at(i));
        edge_end.at(i) = max(p->at(i),edge_end.at(i));
    }
}

long double Minibox::volume ()
{
    long double volume_ = 1;
    for (int i=0; i<(int)edge_begin.size(); i++)
    {
        volume_ = volume_ * (edge_end.at(i) - edge_begin.at(i));
    }
    return (volume_);
}


#define MAX_GROUP_ELEMENTS 50


/*  Hierarchical Clustering classes and functions */


// Hierarchical Clustering linkage variants
enum method_codes
{
    METHOD_METR_SINGLE           = 0,
    METHOD_METR_AVERAGE          = 1
};

#ifndef INT32_MAX
#define MAX_INDEX 0x7fffffffL
#else
#define MAX_INDEX INT32_MAX
#endif

// A node in the hierarchical clustering algorithm
struct node {
    int_fast32_t node1, node2;
    double dist;

    inline friend bool operator< (const node a, const node b)
    {
        // Numbers are always smaller than NaNs.
        return a.dist < b.dist || (a.dist==a.dist && b.dist!=b.dist);
    }
};

// self-destructing array pointer
template <typename type>
class auto_array_ptr {
private:
    type * ptr;
public:
    auto_array_ptr() { ptr = NULL; }
    template <typename index>
    auto_array_ptr(index const size) { init(size); }
    template <typename index, typename value>
    auto_array_ptr(index const size, value const val) { init(size, val); }

    ~auto_array_ptr()
    {
        delete [] ptr;
    }
    void free() {
        delete [] ptr;
        ptr = NULL;
    }
    template <typename index>
    void init(index const size)
    {
        ptr = new type [size];
    }
    template <typename index, typename value>
    void init(index const size, value const val)
    {
        init(size);
        for (index i=0; i<size; i++) ptr[i] = val;
    }
    inline operator type *() const { return ptr; }
};

// The result of the hierarchical clustering algorithm
class cluster_result {
private:
    auto_array_ptr<node> Z;
    int_fast32_t pos;

public:
    cluster_result(const int_fast32_t size): Z(size)
    {
        pos = 0;
    }

    void append(const int_fast32_t node1, const int_fast32_t node2, const double dist)
    {
        Z[pos].node1 = node1;
        Z[pos].node2 = node2;
        Z[pos].dist  = dist;
        pos++;
    }

    node * operator[] (const int_fast32_t idx) const { return Z + idx; }

    void sqrt() const
    {
        for (int_fast32_t i=0; i<pos; i++)
            Z[i].dist = ::sqrt(Z[i].dist);
    }

    void sqrt(const double) const  // ignore the argument
    {
        sqrt();
    }
};

// Class for a doubly linked list
class doubly_linked_list {
public:
    int_fast32_t start;
    auto_array_ptr<int_fast32_t> succ;

private:
    auto_array_ptr<int_fast32_t> pred;

public:
    doubly_linked_list(const int_fast32_t size): succ(size+1), pred(size+1)
    {
        for (int_fast32_t i=0; i<size; i++)
        {
            pred[i+1] = i;
            succ[i] = i+1;
        }
        start = 0;
    }

    void remove(const int_fast32_t idx)
    {
        // Remove an index from the list.
        if (idx==start)
        {
            start = succ[idx];
        } else {
            succ[pred[idx]] = succ[idx];
            pred[succ[idx]] = pred[idx];
        }
        succ[idx] = 0; // Mark as inactive
    }

    bool is_inactive(int_fast32_t idx) const
    {
        return (succ[idx]==0);
    }
};

// Indexing functions
// D is the upper triangular part of a symmetric (NxN)-matrix
// We require r_ < c_ !
#define D_(r_,c_) ( D[(static_cast<std::ptrdiff_t>(2*N-3-(r_))*(r_)>>1)+(c_)-1] )
// Z is an ((N-1)x4)-array
#define Z_(_r, _c) (Z[(_r)*4 + (_c)])

/*
   Lookup function for a union-find data structure.

   The function finds the root of idx by going iteratively through all
   parent elements until a root is found. An element i is a root if
   nodes[i] is zero. To make subsequent searches faster, the entry for
   idx and all its parents is updated with the root element.
*/
class union_find {
private:
    auto_array_ptr<int_fast32_t> parent;
    int_fast32_t nextparent;

public:
    void init(const int_fast32_t size)
    {
        parent.init(2*size-1, 0);
        nextparent = size;
    }

    int_fast32_t Find (int_fast32_t idx) const
    {
        if (parent[idx] !=0 ) // a -> b
        {
            int_fast32_t p = idx;
            idx = parent[idx];
            if (parent[idx] !=0 ) // a -> b -> c
            {
                do
                {
                    idx = parent[idx];
                } while (parent[idx] != 0);
                do
                {
                    int_fast32_t tmp = parent[p];
                    parent[p] = idx;
                    p = tmp;
                } while (parent[p] != idx);
            }
        }
        return idx;
    }

    void Union (const int_fast32_t node1, const int_fast32_t node2)
    {
        parent[node1] = parent[node2] = nextparent++;
    }
};


/* Functions for the update of the dissimilarity array */

inline static void f_single( double * const b, const double a )
{
    if (*b > a) *b = a;
}
inline static void f_average( double * const b, const double a, const double s, const double t)
{
    *b = s*a + t*(*b);
}

/*
     This is the NN-chain algorithm.

     N: integer
     D: condensed distance matrix N*(N-1)/2
     Z2: output data structure
*/
template <const unsigned char method, typename t_members>
static void NN_chain_core(const int_fast32_t N, double * const D, t_members * const members, cluster_result & Z2)
{
    int_fast32_t i;

    auto_array_ptr<int_fast32_t> NN_chain(N);
    int_fast32_t NN_chain_tip = 0;

    int_fast32_t idx1, idx2;

    double size1, size2;
    doubly_linked_list active_nodes(N);

    double min;

    for (int_fast32_t j=0; j<N-1; j++)
    {
        if (NN_chain_tip <= 3)
        {
            NN_chain[0] = idx1 = active_nodes.start;
            NN_chain_tip = 1;

            idx2 = active_nodes.succ[idx1];
            min = D_(idx1,idx2);

            for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
            {
                if (D_(idx1,i) < min)
                {
                    min = D_(idx1,i);
                    idx2 = i;
                }
            }
        }  // a: idx1   b: idx2
        else {
            NN_chain_tip -= 3;
            idx1 = NN_chain[NN_chain_tip-1];
            idx2 = NN_chain[NN_chain_tip];
            min = idx1<idx2 ? D_(idx1,idx2) : D_(idx2,idx1);
        }  // a: idx1   b: idx2

        do {
            NN_chain[NN_chain_tip] = idx2;

            for (i=active_nodes.start; i<idx2; i=active_nodes.succ[i])
            {
                if (D_(i,idx2) < min)
                {
                    min = D_(i,idx2);
                    idx1 = i;
                }
            }
            for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
            {
                if (D_(idx2,i) < min)
                {
                    min = D_(idx2,i);
                    idx1 = i;
                }
            }

            idx2 = idx1;
            idx1 = NN_chain[NN_chain_tip++];

        } while (idx2 != NN_chain[NN_chain_tip-2]);

        Z2.append(idx1, idx2, min);

        if (idx1>idx2)
        {
            int_fast32_t tmp = idx1;
            idx1 = idx2;
            idx2 = tmp;
        }

        //if ( method == METHOD_METR_AVERAGE )
        {
            size1 = static_cast<double>(members[idx1]);
            size2 = static_cast<double>(members[idx2]);
            members[idx2] += members[idx1];
        }

        // Remove the smaller index from the valid indices (active_nodes).
        active_nodes.remove(idx1);

        switch (method) {
            case METHOD_METR_SINGLE:
                /*
                 Single linkage.
                */
                // Update the distance matrix in the range [start, idx1).
                for (i=active_nodes.start; i<idx1; i=active_nodes.succ[i])
                    f_single(&D_(i, idx2), D_(i, idx1) );
                // Update the distance matrix in the range (idx1, idx2).
                for (; i<idx2; i=active_nodes.succ[i])
                    f_single(&D_(i, idx2), D_(idx1, i) );
                // Update the distance matrix in the range (idx2, N).
                for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
                    f_single(&D_(idx2, i), D_(idx1, i) );
                break;

            case METHOD_METR_AVERAGE:
            {
                /*
                Average linkage.
                */
                // Update the distance matrix in the range [start, idx1).
                double s = size1/(size1+size2);
                double t = size2/(size1+size2);
                for (i=active_nodes.start; i<idx1; i=active_nodes.succ[i])
                    f_average(&D_(i, idx2), D_(i, idx1), s, t );
                // Update the distance matrix in the range (idx1, idx2).
                for (; i<idx2; i=active_nodes.succ[i])
                    f_average(&D_(i, idx2), D_(idx1, i), s, t );
                // Update the distance matrix in the range (idx2, N).
                for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
                    f_average(&D_(idx2, i), D_(idx1, i), s, t );
                break;
            }
        }
    }
}


/*
   Clustering methods for vector data
*/

template <typename t_dissimilarity>
static void MST_linkage_core_vector(const int_fast32_t N,
                                    t_dissimilarity & dist,
                                    cluster_result & Z2) {
/*
     Hierarchical clustering using the minimum spanning tree

     N: integer, number of data points
     dist: function pointer to the metric
     Z2: output data structure
*/
    int_fast32_t i;
    int_fast32_t idx2;
    doubly_linked_list active_nodes(N);
    auto_array_ptr<double> d(N);

    int_fast32_t prev_node;
    double min;

    // first iteration
    idx2 = 1;
    min = d[1] = dist(0,1);
    for (i=2; min!=min && i<N; i++) // eliminate NaNs if possible
    {
        min = d[i] = dist(0,i);
        idx2 = i;
    }

    for ( ; i<N; i++)
    {
        d[i] = dist(0,i);
        if (d[i] < min)
        {
            min = d[i];
            idx2 = i;
        }
    }

    Z2.append(0, idx2, min);

    for (int_fast32_t j=1; j<N-1; j++)
    {
        prev_node = idx2;
        active_nodes.remove(prev_node);

        idx2 = active_nodes.succ[0];
        min = d[idx2];

        for (i=idx2; min!=min && i<N; i=active_nodes.succ[i]) // eliminate NaNs if possible
        {
            min = d[i] = dist(i, prev_node);
            idx2 = i;
        }

        for ( ; i<N; i=active_nodes.succ[i])
        {
            double tmp = dist(i, prev_node);
            if (d[i] > tmp)
                d[i] = tmp;
            if (d[i] < min)
            {
                min = d[i];
                idx2 = i;
            }
        }
        Z2.append(prev_node, idx2, min);
    }
}

class linkage_output {
private:
    double * Z;
    int_fast32_t pos;

public:
    linkage_output(double * const _Z)
    {
         this->Z = _Z;
         pos = 0;
    }

    void append(const int_fast32_t node1, const int_fast32_t node2, const double dist, const double size)
    {
         if (node1<node2)
         {
                Z[pos++] = static_cast<double>(node1);
                Z[pos++] = static_cast<double>(node2);
         } else {
                Z[pos++] = static_cast<double>(node2);
                Z[pos++] = static_cast<double>(node1);
         }
         Z[pos++] = dist;
         Z[pos++] = size;
    }
};


/*
    Generate the specific output format for a dendrogram from the
    clustering output.

    The list of merging steps can be sorted or unsorted.
*/

// The size of a node is either 1 (a single point) or is looked up from
// one of the clusters.
#define size_(r_) ( ((r_<N) ? 1 : Z_(r_-N,3)) )

static void generate_dendrogram(double * const Z, cluster_result & Z2, const int_fast32_t N)
{
    // The array "nodes" is a union-find data structure for the cluster
    // identites (only needed for unsorted cluster_result input).
    union_find nodes;
    std::stable_sort(Z2[0], Z2[N-1]);
    nodes.init(N);

    linkage_output output(Z);
    int_fast32_t node1, node2;

    for (int_fast32_t i=0; i<N-1; i++) {
         // Get two data points whose clusters are merged in step i.
         // Find the cluster identifiers for these points.
         node1 = nodes.Find(Z2[i]->node1);
         node2 = nodes.Find(Z2[i]->node2);
         // Merge the nodes in the union-find data structure by making them
         // children of a new node.
         nodes.Union(node1, node2);
         output.append(node1, node2, Z2[i]->dist, size_(node1)+size_(node2));
    }
}

/*
     Clustering on vector data
*/

enum {
    // metrics
    METRIC_EUCLIDEAN       =  0,
    METRIC_CITYBLOCK       =  1,
    METRIC_SEUCLIDEAN      =  2,
    METRIC_SQEUCLIDEAN     =  3
};

/*
    This class handles all the information about the dissimilarity
    computation.
*/
class dissimilarity {
private:
    double * Xa;
    auto_array_ptr<double> Xnew;
    std::ptrdiff_t dim; // size_t saves many statis_cast<> in products
    int_fast32_t N;
    int_fast32_t * members;
    void (cluster_result::*postprocessfn) (const double) const;
    double postprocessarg;

    double (dissimilarity::*distfn) (const int_fast32_t, const int_fast32_t) const;

    auto_array_ptr<double> precomputed;

    double * V;
    const double * V_data;

public:
    dissimilarity (double * const _Xa, int _Num, int _dim,
                   int_fast32_t * const _members,
                   const unsigned char method,
                   const unsigned char metric,
                   bool temp_point_array)
                   : Xa(_Xa),
                     dim(_dim),
                     N(_Num),
                     members(_members),
                     postprocessfn(NULL),
                     V(NULL)
    {
        switch (method) {
            case METHOD_METR_SINGLE: // only single linkage allowed here but others may come...
            default:
                postprocessfn = NULL; // default
                switch (metric)
                {
                    case METRIC_EUCLIDEAN:
                        set_euclidean();
                        break;
                    case METRIC_SEUCLIDEAN:
                    case METRIC_SQEUCLIDEAN:
                        distfn = &dissimilarity::sqeuclidean;
                        break;
                    case METRIC_CITYBLOCK:
                        set_cityblock();
                        break;
                }
        }

        if (temp_point_array)
        {
            Xnew.init((N-1)*dim);
        }
    }

    ~dissimilarity()
    {
        free(V);
    }

    inline double operator () (const int_fast32_t i, const int_fast32_t j) const
    {
        return (this->*distfn)(i,j);
    }

    inline double X (const int_fast32_t i, const int_fast32_t j) const
    {
        return Xa[i*dim+j];
    }

    inline bool Xb (const int_fast32_t i, const int_fast32_t j) const
    {
        return  reinterpret_cast<bool *>(Xa)[i*dim+j];
    }

    inline double * Xptr(const int_fast32_t i, const int_fast32_t j) const
    {
        return Xa+i*dim+j;
    }

    void postprocess(cluster_result & Z2) const
    {
        if (postprocessfn!=NULL)
        {
            (Z2.*postprocessfn)(postprocessarg);
        }
    }

    double sqeuclidean(const int_fast32_t i, const int_fast32_t j) const
    {
        double sum = 0;
        double const * Pi = Xa+i*dim;
        double const * Pj = Xa+j*dim;
        for (int_fast32_t k=0; k<dim; k++)
        {
            double diff = Pi[k] - Pj[k];
            sum += diff*diff;
        }
        return sum;
    }

private:

    void set_euclidean()
    {
        distfn = &dissimilarity::sqeuclidean;
        postprocessfn = &cluster_result::sqrt;
    }

    void set_cityblock()
    {
        distfn = &dissimilarity::cityblock;
    }

    double seuclidean(const int_fast32_t i, const int_fast32_t j) const
    {
        double sum = 0;
        for (int_fast32_t k=0; k<dim; k++)
        {
            double diff = X(i,k)-X(j,k);
            sum += diff*diff/V_data[k];
        }
        return sum;
    }

    double cityblock(const int_fast32_t i, const int_fast32_t j) const
    {
        double sum = 0;
        for (int_fast32_t k=0; k<dim; k++)
        {
            sum += fabs(X(i,k)-X(j,k));
        }
        return sum;
    }
};

/*Clustering for the "stored data approach": the input are points in a vector space.*/
static int linkage_vector(double *X, int N, int dim, double * Z, unsigned char method, unsigned char metric)
{

    CV_Assert(N >=1);
    CV_Assert(N <= MAX_INDEX/4);
    CV_Assert(dim >=1);

    try
    {
        cluster_result Z2(N-1);
        auto_array_ptr<int_fast32_t> members;
        dissimilarity dist(X, N, dim, members, method, metric, false);
        MST_linkage_core_vector(N, dist, Z2);
        dist.postprocess(Z2);
        generate_dendrogram(Z, Z2, N);
    } // try
    catch (const std::bad_alloc&)
    {
        CV_Error(CV_StsNoMem, "Not enough Memory for erGrouping hierarchical clustering structures!");
    }
    catch(const std::exception&)
    {
        CV_Error(CV_StsError, "Uncaught exception in erGrouping!");
    }
    catch(...)
    {
        CV_Error(CV_StsError, "C++ exception (unknown reason) in erGrouping!");
    }
    return 0;
}

// ERFeatures structure stores additional features for a given ERStat instance
struct ERFeatures
{
    int area;
    Point center;
    Rect  rect;
    float intensity_mean;  ///< mean intensity of the whole region
    float intensity_std;  ///< intensity standard deviation of the whole region
    float boundary_intensity_mean;  ///< mean intensity of the boundary of the region
    float boundary_intensity_std;  ///< intensity standard deviation of the boundary of the region
    double stroke_mean;  ///< mean stroke width approximation of the whole region
    double stroke_std;  ///< stroke standard deviation of the whole region
    double gradient_mean;  ///< mean gradient magnitude of the whole region
    double gradient_std;  ///< gradient magnitude standard deviation of the whole region
    double axial_ratio;
    double convex_hull_ratio;
    int convexities;
    double hu_moments[7];
};

/*  Maximal Meaningful Clusters Detection */

struct HCluster{
    int num_elem;           // number of elements
    vector<int> elements;   // elements (contour ID)
    int nfa;                // the number of false alarms for this merge
    float dist;             // distance of the merge
    float dist_ext;         // distamce where this merge will merge with another
    long double volume;     // volume of the bounding sphere (or bounding box)
    long double volume_ext; // volume of the sphere(or box) + envolvent empty space
    vector<vector<float> > points; // nD points in this cluster
    bool max_meaningful;    // is this merge max meaningul ?
    vector<int> max_in_branch; // otherwise which merges are the max_meaningful in this branch
    int min_nfa_in_branch;  // min nfa detected within the chilhood
    int node1;
    int node2;
    double probability;      //the probability of this group of being a text group
};

class MaxMeaningfulClustering
{
public:
    unsigned char method_;
    unsigned char metric_;

    /// Constructor.
    MaxMeaningfulClustering(unsigned char _method, unsigned char _metric, vector<ERFeatures> &_regions,
                            Size _imsize, const std::string &filename, double _minProbability);

    void operator()(double *data, unsigned int num, int dim, unsigned char method,
                    unsigned char metric, vector< vector<int> > *meaningful_clusters);

private:
    double minProbability;
    CvBoost group_boost;
    vector<ERFeatures> &regions;
    Size imsize;

    /// Helper functions
    void build_merge_info(double *dendogram, double *data, int num, int dim, bool use_full_merge_rule,
                          vector<HCluster> *merge_info, vector< vector<int> > *meaningful_clusters);

    /// Calculate the Number of False Alarms
    int nfa(float sigma, int k, int N);

    /// Calculate the probability of a group being a text group
    double probability(vector<int> &elements);

};

MaxMeaningfulClustering::MaxMeaningfulClustering(unsigned char _method, unsigned char _metric, vector<ERFeatures> &_regions,
                                                 Size _imsize, const std::string &filename, double _minProbability):
                                                 method_(_method), metric_(_metric), regions(_regions), imsize(_imsize)
{

    minProbability = _minProbability;

    if (ifstream(filename.c_str()))
        group_boost.load( filename.c_str(), "boost" );
    else
        CV_Error(CV_StsBadArg, "erGrouping: Default classifier file not found!");
}


void MaxMeaningfulClustering::operator()(double *data, unsigned int num, int dim, unsigned char method,
                                         unsigned char metric, vector< vector<int> > *meaningful_clusters)
{

    double *Z = (double*)malloc(((num-1)*4) * sizeof(double)); // we need 4 floats foreach sample merge.
    if (Z == NULL)
        CV_Error(CV_StsNoMem, "Not enough Memory for erGrouping hierarchical clustering structures!");

    linkage_vector(data, (int)num, dim, Z, method, metric);

    vector<HCluster> merge_info;
    build_merge_info(Z, data, (int)num, dim, false, &merge_info, meaningful_clusters);

    free(Z);
    merge_info.clear();
}

void MaxMeaningfulClustering::build_merge_info(double *Z, double *X, int N, int dim,
                                               bool use_full_merge_rule,
                                               vector<HCluster> *merge_info,
                                               vector< vector<int> > *meaningful_clusters)
{

    // walk the whole dendogram
    for (int i=0; i<(N-1)*4; i=i+4)
    {
        HCluster cluster;
        cluster.num_elem = (int)Z[i+3]; //number of elements

        int node1  = (int)Z[i];
        int node2  = (int)Z[i+1];
        float dist = (float)Z[i+2];

        if (node1<N)
        {
            vector<float> point;
            for (int n=0; n<dim; n++)
                point.push_back((float)X[node1*dim+n]);
            cluster.points.push_back(point);
            cluster.elements.push_back((int)node1);
        }
        else
        {
            for (int ii=0; ii<(int)merge_info->at(node1-N).points.size(); ii++)
            {
                cluster.points.push_back(merge_info->at(node1-N).points[ii]);
                cluster.elements.push_back(merge_info->at(node1-N).elements[ii]);
            }
            //update the extended volume of node1 using the dist where this cluster merge with another
            merge_info->at(node1-N).dist_ext = dist;
        }
        if (node2<N)
        {
            vector<float> point;
            for (int n=0; n<dim; n++)
                point.push_back((float)X[node2*dim+n]);
            cluster.points.push_back(point);
            cluster.elements.push_back((int)node2);
        }
        else
        {
            for (int ii=0; ii<(int)merge_info->at(node2-N).points.size(); ii++)
            {
                cluster.points.push_back(merge_info->at(node2-N).points[ii]);
                cluster.elements.push_back(merge_info->at(node2-N).elements[ii]);
            }

            //update the extended volume of node2 using the dist where this cluster merge with another
            merge_info->at(node2-N).dist_ext = dist;
        }

        Minibox mb;
        for (int ii=0; ii<(int)cluster.points.size(); ii++)
        {
            mb.check_in(&cluster.points.at(ii));
        }

        cluster.dist   = dist;
        cluster.volume = mb.volume();
        if (cluster.volume >= 1)
            cluster.volume = 0.999999;
        if (cluster.volume == 0)
            cluster.volume = 0.001;

        cluster.volume_ext=1;

        if (node1>=N)
        {
            merge_info->at(node1-N).volume_ext = cluster.volume;
        }
        if (node2>=N)
        {
            merge_info->at(node2-N).volume_ext = cluster.volume;
        }

        cluster.node1 = node1;
        cluster.node2 = node2;

        merge_info->push_back(cluster);

    }

    for (int i=0; i<(int)merge_info->size(); i++)
    {

        merge_info->at(i).nfa = nfa((float)merge_info->at(i).volume,
                                    merge_info->at(i).num_elem, N);

        merge_info->at(i).probability = probability(merge_info->at(i).elements);
        int node1 = merge_info->at(i).node1;
        int node2 = merge_info->at(i).node2;

        if ((node1<N)&&(node2<N))
        {
            // both nodes are individual samples (nfa=1) : each cluster is max.
            merge_info->at(i).max_meaningful = true;
            merge_info->at(i).max_in_branch.push_back(i);
            merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
        } else {
            if ((node1>=N)&&(node2>=N))
            {
                // both nodes are "sets" : we must evaluate the merging condition
                if ( ( ( (use_full_merge_rule) &&
                         ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + merge_info->at(node2-N).nfa) &&
                         (merge_info->at(i).nfa < min(merge_info->at(node1-N).min_nfa_in_branch,
                                                    merge_info->at(node2-N).min_nfa_in_branch))) ) ||
                       ( (!use_full_merge_rule) &&
                         ((merge_info->at(i).nfa < min(merge_info->at(node1-N).min_nfa_in_branch,
                                                     merge_info->at(node2-N).min_nfa_in_branch))) ) )
                     && (merge_info->at(i).probability > minProbability) )
                {
                    merge_info->at(i).max_meaningful = true;
                    merge_info->at(i).max_in_branch.push_back(i);
                    merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                    for (int k =0; k<(int)merge_info->at(node1-N).max_in_branch.size(); k++)
                        merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;
                    for (int k =0; k<(int)merge_info->at(node2-N).max_in_branch.size(); k++)
                        merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
                } else {
                    merge_info->at(i).max_meaningful = false;
                    merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                                                           merge_info->at(node1-N).max_in_branch.begin(),
                                                           merge_info->at(node1-N).max_in_branch.end());
                    merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                                                           merge_info->at(node2-N).max_in_branch.begin(),
                                                           merge_info->at(node2-N).max_in_branch.end());

                    if (merge_info->at(i).nfa < min(merge_info->at(node1-N).min_nfa_in_branch,
                                                    merge_info->at(node2-N).min_nfa_in_branch))

                        merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                    else
                        merge_info->at(i).min_nfa_in_branch = min(merge_info->at(node1-N).min_nfa_in_branch,
                                                                  merge_info->at(node2-N).min_nfa_in_branch);
                }
            } else {

                //one of the nodes is a "set" and the other is an individual sample : check merging condition
                if (node1>=N)
                {
                    if ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + 1) &&
                        (merge_info->at(i).nfa<merge_info->at(node1-N).min_nfa_in_branch) &&
                        (merge_info->at(i).probability > minProbability))
                    {
                        merge_info->at(i).max_meaningful = true;
                        merge_info->at(i).max_in_branch.push_back(i);
                        merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                        for (int k =0; k<(int)merge_info->at(node1-N).max_in_branch.size(); k++)
                            merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;
                    } else {
                        merge_info->at(i).max_meaningful = false;
                        merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                                                               merge_info->at(node1-N).max_in_branch.begin(),
                                                               merge_info->at(node1-N).max_in_branch.end());
                        merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,
                                                                  merge_info->at(node1-N).min_nfa_in_branch);
                    }
                } else {
                    if ((merge_info->at(i).nfa < merge_info->at(node2-N).nfa + 1) &&
                        (merge_info->at(i).nfa<merge_info->at(node2-N).min_nfa_in_branch) &&
                        (merge_info->at(i).probability > minProbability))
                    {
                        merge_info->at(i).max_meaningful = true;
                        merge_info->at(i).max_in_branch.push_back(i);
                        merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                        for (int k =0; k<(int)merge_info->at(node2-N).max_in_branch.size(); k++)
                            merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
                    } else {
                        merge_info->at(i).max_meaningful = false;
                        merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                        merge_info->at(node2-N).max_in_branch.begin(),
                        merge_info->at(node2-N).max_in_branch.end());
                        merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,
                        merge_info->at(node2-N).min_nfa_in_branch);
                    }
                }
            }
        }
    }

    for (int i=0; i<(int)merge_info->size(); i++)
    {
        if (merge_info->at(i).max_meaningful)
        {
            vector<int> cluster;
            for (int k=0; k<(int)merge_info->at(i).elements.size();k++)
                cluster.push_back(merge_info->at(i).elements.at(k));
            meaningful_clusters->push_back(cluster);
        }
    }

}


int MaxMeaningfulClustering::nfa(float sigma, int k, int N)
{
    // use an approximation for the nfa calculations (faster)
    return -1*(int)NFA( N, k, (double) sigma, 0);
}


// utility functions for MST
static bool edge_comp (Vec4f i,Vec4f j)
{
    Point a = Point(cvRound(i[0]), cvRound(i[1]));
    Point b = Point(cvRound(i[2]), cvRound(i[3]));
    double edist_i = cv::norm(a-b);
    a = Point(cvRound(j[0]), cvRound(j[1]));
    b = Point(cvRound(j[2]), cvRound(j[3]));
    double edist_j = cv::norm(a-b);
    return (edist_i>edist_j);
}

static int find_vertex(vector<vector<Point> > &forest, Point &p)
{
  for (int i=0; i<(int)forest.size(); i++)
  {
    for (int j=0; j<(int)forest[i].size(); j++)
    {
        if (forest[i][j] == p)
            return i;
    }
  }
  return -1;
}

static int getAngleABC( Point a, Point b, Point c )
{
    Point ab = Point( b.x - a.x, b.y - a.y );
    Point cb = Point( b.x - c.x, b.y - c.y );

    // dot product
    float dot = (ab.x * cb.x + ab.y * cb.y);

    // length square of both vectors
    float abSqr = ab.x * ab.x + ab.y * ab.y;
    float cbSqr = cb.x * cb.x + cb.y * cb.y;

    // square of cosine of the needed angle
    float cosSqr = dot * dot / abSqr / cbSqr;

    // this is a known trigonometric equality:
    // cos(alpha * 2) = [ cos(alpha) ]^2 * 2 - 1
    float cos2 = 2 * cosSqr - 1;

    // Here's the only invocation of the heavy function.
    // It's a good idea to check explicitly if cos2 is within [-1 .. 1] range

    const float pi = 3.141592f;

    float alpha2 =
        (cos2 <= -1) ? pi :
        (cos2 >= 1) ? 0 :
        acosf(cos2);

    float rslt = alpha2 / 2;

    float rs = rslt * 180. / pi;


    // Now revolve the ambiguities.
    // 1. If dot product of two vectors is negative - the angle is definitely
    // above 90 degrees. Still we have no information regarding the sign of the angle.

    // NOTE: This ambiguity is the consequence of our method: calculating the cosine
    // of the double angle. This allows us to get rid of calling sqrt.

    if (dot < 0)
        rs = 180 - rs;

    // 2. Determine the sign. For this we'll use the Determinant of two vectors.
    float det = (ab.x * cb.y - ab.y * cb.y);
    if (det < 0)
        rs = -rs;

    return abs((int) floor(rs + 0.5));
}

double MaxMeaningfulClustering::probability(vector<int> &cluster)
{

    if (cluster.size()>MAX_GROUP_ELEMENTS)
        return 0.;

    vector<float> sample;
    sample.push_back(0);
    sample.push_back(cluster.size());

    Mat diameters      ( cluster.size(), 1, CV_32F, 1 );
    Mat strokes        ( cluster.size(), 1, CV_32F, 1 );
    Mat gradients      ( cluster.size(), 1, CV_32F, 1 );
    Mat fg_intensities ( cluster.size(), 1, CV_32F, 1 );
    Mat bg_intensities ( cluster.size(), 1, CV_32F, 1 );
    Mat axial_ratios   ( cluster.size(), 1, CV_32F, 1 );
    Mat chull_ratios   ( cluster.size(), 1, CV_32F, 1 );
    Mat convexities    ( cluster.size(), 1, CV_32F, 1 );
    Subdiv2D subdiv(Rect(0,0,imsize.width,imsize.height));
    vector< vector<Point> > forest(cluster.size());
    float maxAvgOverlap = 0;

    for (int i=(int)cluster.size()-1; i>=0; i--)
    {

        diameters.at<float>(i,0)      = (float)max(regions.at(cluster.at(i)).rect.width,regions.at(cluster.at(i)).rect.height);
        strokes.at<float>(i,0)        = (float)regions.at(cluster.at(i)).stroke_mean;
        gradients.at<float>(i,0)      = (float)regions.at(cluster.at(i)).gradient_mean;
        fg_intensities.at<float>(i,0) = (float)regions.at(cluster.at(i)).intensity_mean;
        bg_intensities.at<float>(i,0) = (float)regions.at(cluster.at(i)).boundary_intensity_mean;
        axial_ratios.at<float>(i,0)   = (float)regions.at(cluster.at(i)).axial_ratio;
        chull_ratios.at<float>(i,0)   = (float)regions.at(cluster.at(i)).convex_hull_ratio;
        convexities.at<float>(i,0)    = (float)regions.at(cluster.at(i)).convexities;

        Point2f fp((float)regions.at(cluster.at(i)).rect.x+(regions.at(cluster.at(i)).rect.width/2),
                   (float)regions.at(cluster.at(i)).rect.y+(regions.at(cluster.at(i)).rect.height/2));
        subdiv.insert(fp);
        forest[i].push_back(Point((int)fp.x,(int)fp.y));
        float avgOverlap = 0;
        for (int j=0; j<(int)cluster.size(); j++)
        {
            if (j!=i)
            {
                Rect intersection = regions.at(cluster.at(i)).rect & regions.at(cluster.at(j)).rect;
                int area_intersection = intersection.width * intersection.height;
                int area_i = regions.at(cluster.at(i)).rect.width * regions.at(cluster.at(i)).rect.height;
                int area_j = regions.at(cluster.at(j)).rect.width * regions.at(cluster.at(j)).rect.height;
                if (area_intersection > 0)
                {
                    float overlap = (float)area_intersection / min(area_i, area_j);
                    avgOverlap += overlap;
                }
            }
        }
        avgOverlap = avgOverlap / (cluster.size()-1);
        if (avgOverlap > maxAvgOverlap)
            maxAvgOverlap = avgOverlap;
    }

    Scalar mean,std;
    meanStdDev( diameters, mean, std );
    sample.push_back(std[0]/mean[0]); float diameter_mean = mean[0];
    meanStdDev( strokes, mean, std );
    sample.push_back(std[0]/mean[0]);
    meanStdDev( gradients, mean, std );
    sample.push_back(std[0]);
    meanStdDev( fg_intensities, mean, std );
    sample.push_back(std[0]);
    meanStdDev( bg_intensities, mean, std );
    sample.push_back(std[0]);

    /* begin Kruskal algorithm to find the MST */
    vector<Vec4f> edgeList;
    subdiv.getEdgeList(edgeList);
    std::sort (edgeList.begin(), edgeList.end(), edge_comp);

    vector<Vec4f> mst_edges;
    vector<float> edge_distances;

    for( int k = (int)edgeList.size()-1; k>=0; k-- )
    {
        Vec4f e = edgeList[k];
        Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
        Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
        int tree_pt0  = find_vertex(forest, pt0);
        int tree_pt1  = find_vertex(forest, pt1);
        if (((pt0.x>0)&&(pt0.x<imsize.width)&&(pt0.y>0)&&(pt0.y<imsize.height) &&
             (pt1.x>0)&&(pt1.x<imsize.width)&&(pt1.y>0)&&(pt1.y<imsize.height)) &&
                (tree_pt0 != tree_pt1))
        {
            mst_edges.push_back(e);
            forest[tree_pt0].insert(forest[tree_pt0].begin(),forest[tree_pt1].begin(),forest[tree_pt1].end());
            forest.erase(forest.begin()+tree_pt1);
            edge_distances.push_back((float)norm(pt0-pt1));
        }
        if (mst_edges.size() == cluster.size()-1)
            break;
    }

    //cout << "mst has " << mst_edges.size() << " edges " << endl;

    /* End Kruskal algorithm */

    vector<float> angles;
    for (size_t k=0; k<mst_edges.size(); k++)
    {
        Vec4f q = mst_edges[k];
        Point q_pt0 = Point(cvRound(q[0]), cvRound(q[1]));
        Point q_pt1 = Point(cvRound(q[2]), cvRound(q[3]));
        for (size_t j=k+1; j<mst_edges.size(); j++)
        {
            Vec4f t = mst_edges[j];
            Point t_pt0 = Point(cvRound(t[0]), cvRound(t[1]));
            Point t_pt1 = Point(cvRound(t[2]), cvRound(t[3]));
            if(q_pt0 == t_pt0)
                angles.push_back(getAngleABC(q_pt1, q_pt0 , t_pt1));
            if(q_pt0 == t_pt1)
                angles.push_back(getAngleABC(q_pt1, q_pt0 , t_pt0));
            if(q_pt1 == t_pt0)
                angles.push_back(getAngleABC(q_pt0, q_pt1 , t_pt1));
            if(q_pt1 == t_pt1)
                angles.push_back(getAngleABC(q_pt0, q_pt1 , t_pt0));
        }
    }
    //cout << "we have " << angles.size() << " angles " << endl;
    //for (int kk=0; kk<angles.size(); kk++)
    //  cout << angles[kk] << " ";
    //cout << endl;

    meanStdDev( angles, mean, std );
    sample.push_back(std[0]);
    sample.push_back(mean[0]);
    meanStdDev( edge_distances, mean, std );
    sample.push_back(std[0]/mean[0]);
    sample.push_back(mean[0]/diameter_mean);

    meanStdDev( axial_ratios, mean, std );
    sample.push_back(mean[0]);
    sample.push_back(std[0]);

    /// Calculate average shape self-similarity
    double avg_shape_match = 0;
    double eps = 1.e-5;
    int num_matches = 0, sma, smb;
    for (size_t i=0; i<cluster.size(); i++)
    {
        for (size_t j=i+1; j<cluster.size(); j++)
        {
            for (int h=0; h<7; h++)
            {
                double ama = fabs( regions[cluster[i]].hu_moments[h] );
                double amb = fabs( regions[cluster[j]].hu_moments[h] );

                if( regions[cluster[i]].hu_moments[h] > 0 )
                    sma = 1;
                else if( regions[cluster[i]].hu_moments[h] < 0 )
                    sma = -1;
                else
                    sma = 0;
                if( regions[cluster[j]].hu_moments[h] > 0 )
                    smb = 1;
                else if( regions[cluster[j]].hu_moments[h] < 0 )
                    smb = -1;
                else
                    smb = 0;

                if( ama > eps && amb > eps )
                {
                    ama = 1. / (sma * log10( ama ));
                    amb = 1. / (smb * log10( amb ));
                    avg_shape_match += fabs( -ama + amb );
                }
            }
            num_matches++;
        }
    }

    sample.push_back((float)(avg_shape_match/num_matches));

    sample.push_back(maxAvgOverlap);

    meanStdDev( chull_ratios, mean, std );
    sample.push_back(mean[0]);
    sample.push_back(std[0]);

    meanStdDev( convexities, mean, std );
    sample.push_back(mean[0]);
    sample.push_back(std[0]);

    float votes_group = group_boost.predict( Mat(sample), Mat(), Range::all(), false, true );

    return (double)1-(double)1/(1+exp(-2*votes_group));
}


/* fast thinning for stroke width calculation */
bool guo_hall_thinning(const cv::Mat1b & img, cv::Mat& skeleton);

bool guo_hall_thinning(const cv::Mat1b & img, cv::Mat& skeleton)
{

  uchar* img_ptr = img.data;
  uchar* skel_ptr = skeleton.data;

  for (int row = 0; row < img.rows; ++row)
  {
    for (int col = 0; col < img.cols; ++col)
    {
      if (*img_ptr)
      {
        int key = row * img.cols + col;
        if ((col > 0 && *img_ptr != img.data[key - 1]) ||
            (col < img.cols-1 && *img_ptr != img.data[key + 1]) ||
            (row > 0 && *img_ptr != img.data[key - img.cols]) ||
            (row < img.rows-1 && *img_ptr != img.data[key + img.cols]))
        {
          *skel_ptr = 255;
        }
        else
        {
          *skel_ptr = 128;
        }
      }
      img_ptr++;
      skel_ptr++;
    }
  }

  int max_iters = 10000;
  int niters = 0;
  bool changed = false;

  /// list of keys to set to 0 at the end of the iteration
  std::deque<int> cols_to_set;
  std::deque<int> rows_to_set;

  while (changed && niters < max_iters)
  {
    changed = false;
    for (unsigned short iter = 0; iter < 2; ++iter)
    {
      uchar *skeleton_ptr = skeleton.data;
      rows_to_set.clear();
      cols_to_set.clear();
      // for each point in skeleton, check if it needs to be changed
      for (int row = 0; row < skeleton.rows; ++row)
      {
        for (int col = 0; col < skeleton.cols; ++col)
        {
          if (*skeleton_ptr++ == 255)
          {
            bool p2, p3, p4, p5, p6, p7, p8, p9;
            p2 = skeleton.data[(row-1) * skeleton.cols + col];
            p3 = skeleton.data[(row-1) * skeleton.cols + col+1];
            p4 = skeleton.data[row     * skeleton.cols + col+1];
            p5 = skeleton.data[(row+1) * skeleton.cols + col+1];
            p6 = skeleton.data[(row+1) * skeleton.cols + col];
            p7 = skeleton.data[(row+1) * skeleton.cols + col-1];
            p8 = skeleton.data[row     * skeleton.cols + col-1];
            p9 = skeleton.data[(row-1) * skeleton.cols + col-1];

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                    (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

            if ((C == 1) && (N >= 2) && (N <= 3) && (m == 0))
            {
              cols_to_set.push_back(col);
              rows_to_set.push_back(row);
            }
          }
        }
      }

      // set all points in rows_to_set (of skel)
      unsigned int rows_to_set_size = rows_to_set.size();
      for (unsigned int pt_idx = 0; pt_idx < rows_to_set_size; ++pt_idx)
      {
        if (!changed)
          changed = (skeleton.data[rows_to_set[pt_idx] * skeleton.cols + cols_to_set[pt_idx]]);

        int key = rows_to_set[pt_idx] * skeleton.cols + cols_to_set[pt_idx];
        skeleton.data[key] = 0;
        if (cols_to_set[pt_idx] > 0 && skeleton.data[key - 1] == 128) // left
            skeleton.data[key - 1] = 255;
        if (cols_to_set[pt_idx] < skeleton.cols-1 && skeleton.data[key + 1] == 128) // right
            skeleton.data[key + 1] = 255;
        if (rows_to_set[pt_idx] > 0 && skeleton.data[key - skeleton.cols] == 128) // up
            skeleton.data[key - skeleton.cols] = 255;
        if (rows_to_set[pt_idx] < skeleton.rows-1 && skeleton.data[key + skeleton.cols] == 128) // down
            skeleton.data[key + skeleton.cols] = 255;
      }

      if ((niters++) >= max_iters) // we have done!
        break;
    }
  }

  skeleton = (skeleton != 0);
  return true;
}


float extract_features(Mat &grey, Mat& channel, vector<ERStat> &regions, vector<ERFeatures> &features);

float extract_features(Mat &grey, Mat& channel, vector<ERStat> &regions, vector<ERFeatures> &features)
{
    // assert correct image type
    CV_Assert(( channel.type() == CV_8UC1 ) && ( grey.type() == CV_8UC1 ));
    CV_Assert( channel.size() == grey.size() );

    CV_Assert( !regions.empty() );
    CV_Assert( features.empty() );

    Mat gradient_magnitude = Mat_<double>(grey.size());
    get_gradient_magnitude( grey, gradient_magnitude);

    Mat region_mask = Mat::zeros(grey.rows+2, grey.cols+2, CV_8UC1);

    float max_stroke = 0;

    for (int r=0; r<(int)regions.size(); r++)
    {
        ERFeatures f;
        ERStat *stat = &regions.at(r);

        f.area = stat->area;
        f.rect = stat->rect;
        f.center = Point(f.rect.x+(f.rect.width/2),f.rect.y+(f.rect.height/2));

        if (regions.at(r).parent != NULL)
        {

            //Fill the region and calculate features
            Mat region = region_mask(Rect(Point(stat->rect.x,stat->rect.y),
                                          Point(stat->rect.br().x+2,stat->rect.br().y+2)));
            region = Scalar(0);
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            Rect rect;

            floodFill( channel(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x,stat->rect.br().y))),
                       region, Point(stat->pixel%channel.cols - stat->rect.x, stat->pixel/channel.cols - stat->rect.y),
                       Scalar(255), &rect, Scalar(stat->level), Scalar(0), flags );
            rect.width += 2;
            rect.height += 2;
            Mat rect_mask = region_mask(Rect(stat->rect.x+1,stat->rect.y+1,stat->rect.width,stat->rect.height));


            Scalar mean,std;
            meanStdDev( grey(stat->rect), mean, std, rect_mask);
            f.intensity_mean = (float)mean[0];
            f.intensity_std  = (float)std[0];

            Mat tmp,bw;
            region_mask(Rect(stat->rect.x+1,stat->rect.y+1,stat->rect.width,stat->rect.height)).copyTo(bw);
            distanceTransform(bw, tmp, DIST_L1,3); //L1 gives distance in round integers while L2 floats

            // Add border because if region span all the image size skeleton will crash
            copyMakeBorder(bw, bw, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));
            Mat skeleton = Mat::zeros(bw.size(),CV_8UC1);
            guo_hall_thinning(bw,skeleton);
            Mat mask;
            skeleton(Rect(5,5,bw.cols-10,bw.rows-10)).copyTo(mask);
            bw(Rect(5,5,bw.cols-10,bw.rows-10)).copyTo(bw);
            meanStdDev(tmp,mean,std,mask);
            f.stroke_mean = mean[0];
            f.stroke_std  = std[0];

            if (f.stroke_mean > max_stroke)
                max_stroke = (float)f.stroke_mean;

            Mat element = getStructuringElement( MORPH_RECT, Size(5, 5), Point(2, 2) );
            dilate(rect_mask, tmp, element);
            absdiff(tmp, rect_mask, tmp);

            meanStdDev( grey(stat->rect), mean, std, tmp);
            f.boundary_intensity_mean = (float)mean[0];
            f.boundary_intensity_std  = (float)std[0];

            Mat tmp2;
            dilate(rect_mask, tmp, element);
            erode (rect_mask, tmp2, element);
            absdiff(tmp, tmp2, tmp);

            meanStdDev( gradient_magnitude(stat->rect), mean, std, tmp);
            f.gradient_mean = mean[0];
            f.gradient_std  = std[0];

            copyMakeBorder(bw, bw, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));

            vector<vector<Point> > contours0;
            vector<Vec4i> hierarchy;
            findContours( bw, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            RotatedRect rrect = minAreaRect(contours0.at(0));

            f.axial_ratio = max(rrect.size.width, rrect.size.height) / min(rrect.size.width, rrect.size.height);

            Moments mu = moments(contours0.at(0));
            HuMoments (mu, f.hu_moments);

            vector<Point> hull;
            convexHull(contours0[0],hull);
            f.convex_hull_ratio = (float)contourArea(hull)/contourArea(contours0[0]);
            vector<Vec4i> cx;
            vector<int> hull_idx;
            //TODO check epsilon parameter of approxPolyDP (set empirically) : we want more precission
            //     if the region is very small because otherwise we'll loose all the convexities
            approxPolyDP( Mat(contours0[0]), contours0[0], (float)min(rrect.size.width,rrect.size.height)/17, true );
            convexHull(contours0[0],hull_idx,false,false);
            f.convexities = 0;
            if (hull_idx.size()>2)
                if (contours0[0].size()>3)
                    convexityDefects(contours0[0],hull_idx,cx);
            f.convexities = cx.size();

            rect_mask = Scalar(0);

        } else {

            f.intensity_mean = 0;
            f.intensity_std  = 0;

            f.stroke_mean = 0;
            f.stroke_std  = 0;

            f.boundary_intensity_mean = 0;
            f.boundary_intensity_std  = 0;

            f.gradient_mean = 0;
            f.gradient_std  = 0;
        }

        features.push_back(f);
    }

    return max_stroke;
}


/*!
    Find groups of Extremal Regions that are organized as text blocks. This function implements
    the grouping algorithm described in:
    Gomez L. and Karatzas D.: Multi-script Text Extraction from Natural Scenes, ICDAR 2013.
    Notice that this implementation constrains the results to horizontally-aligned text and
    latin script (since ERFilter default classifiers are trained only for latin script detection).

    The algorithm combines two different clustering techniques in a single parameter-free procedure
    to detect groups of regions organized as text. The maximally meaningful groups are fist detected
    in several feature spaces, where each feature space is a combination of proximity information
    (x,y coordinates) and a similarity measure (intensity, color, size, gradient magnitude, etc.),
    thus providing a set of hypotheses of text groups. Evidence Accumulation framework is used to
    combine all these hypotheses to get the final estimate. Each of the resulting groups are finally
    validated by a classifier in order to assest if they form a valid horizontally-aligned text block.

    \param  src            Vector of sinle channel images CV_8UC1 from wich the regions were extracted.
    \param  regions        Vector of ER's retreived from the ERFilter algorithm from each channel
    \param  filename       The XML or YAML file with the classifier model (e.g. trained_classifier_erGrouping.xml)
    \param  minProbability The minimum probability for accepting a group
    \param  groups         The output of the algorithm are stored in this parameter as list of rectangles.
*/
static void erGroupingGK(InputArray _image, InputArrayOfArrays _src, vector<vector<ERStat> > &regions, std::vector<std::vector<Vec2i> > &groups,  std::vector<Rect> &text_boxes, const std::string& filename, float minProbability)
{

    CV_Assert( _image.getMat().type() == CV_8UC3 );
    // TODO assert correct vector<Mat>

    Mat image = _image.getMat();
    Mat grey;
    cvtColor(image, grey, COLOR_BGR2GRAY);

    std::vector<Mat> src;
    _src.getMatVector(src);

    CV_Assert ( !src.empty() );
    CV_Assert ( src.size() == regions.size() );

    if (!text_boxes.empty())
    {
        text_boxes.clear();
    }

    for (int c=0; c<(int)src.size(); c++)
    {
        Mat channel = src.at(c);

        // assert correct image type
        CV_Assert( channel.type() == CV_8UC1 );

        CV_Assert( !regions.at(c).empty() );

        if ( regions.at(c).size() < 3 )
            continue;


        std::vector<vector<int> > meaningful_clusters;
        vector<ERFeatures> features;
        float max_stroke = extract_features(grey, channel, regions.at(c), features);



        // Find the Max. Meaningful Clusters in the learned feature space

        unsigned int N = (unsigned int)regions.at(c).size();
        int dim = 7; //dimensionality of feature space
        double *data = (double*)malloc(dim*N * sizeof(double));
        if (data == NULL)
            CV_Error(CV_StsNoMem, "Not enough Memory for erGrouping hierarchical clustering structures!");

        //Learned weights
        float weight_param1 = 1.00f;
        float weight_param2 = 0.65f;
        float weight_param3 = 0.65f;
        float weight_param4 = 0.49f;
        float weight_param5 = 0.67f;
        float weight_param6 = 0.91f;

        int count = 0;
        for (int i=0; i<(int)regions.at(c).size(); i++)
        {
            data[count] = (double)features.at(i).center.x/channel.cols*weight_param1;
            data[count+1] = (double)features.at(i).center.y/channel.rows*weight_param1;
            data[count+2] = (double)features.at(i).intensity_mean/255*weight_param2;
            data[count+3] = (double)features.at(i).boundary_intensity_mean/255*weight_param3;
            data[count+4] = (double)max(features.at(i).rect.height,features.at(i).rect.width)/
                                    max(channel.rows,channel.cols)*weight_param5;
            data[count+5] = (double)features.at(i).stroke_mean/max_stroke*weight_param6;
            data[count+6] = (double)features.at(i).gradient_mean/255*weight_param4;

            count = count+dim;
        }

        MaxMeaningfulClustering   mm_clustering(METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, features, Size(channel.cols,channel.rows), filename, minProbability);
        mm_clustering(data, N, dim, METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, &meaningful_clusters);

        free(data);

        for (size_t k=0; k<meaningful_clusters.size(); k++)
        {
            if (meaningful_clusters[k].size()>2)
            {
                Rect group_rect = features[meaningful_clusters[k][0]].rect;
                vector<Vec2i> group;
                group.push_back(Vec2i(c,meaningful_clusters[k][0]));
                for (size_t l=1; l<meaningful_clusters[k].size(); l++)
                {
                    group_rect = group_rect | features[meaningful_clusters[k][l]].rect;
                    group.push_back(Vec2i(c,meaningful_clusters[k][l]));
                }
                text_boxes.push_back(group_rect);
                groups.push_back(group);
            }
        }

        //getLines(img, &regions, &final_clusters, line_rects, line_regions, multi_oriented);

        meaningful_clusters.clear();
        features.clear();
    }
}

void erGrouping(InputArray image, InputArrayOfArrays channels, std::vector<std::vector<ERStat> > &regions,  std::vector<std::vector<Vec2i> > &groups,  std::vector<Rect> &groups_rects, int method, const std::string& filename, float minProbability)
{
    CV_Assert( image.getMat().type() == CV_8UC3 );
    CV_Assert( !channels.empty() );
    CV_Assert( !((method == ERGROUPING_ORIENTATION_ANY) && (filename.empty())) );

    switch (method)
    {
        case ERGROUPING_ORIENTATION_HORIZ:
            break;
        case ERGROUPING_ORIENTATION_ANY:
            erGroupingGK(image, channels, regions, groups, groups_rects, filename, minProbability);
            break;
    }

}
}
