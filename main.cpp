
// opencv 
#include <opencv2/opencv.hpp>  //warpPerspective
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp> // findHomography
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

// lib
//#include "opticalFlow.h"
#include "ransac.h"

// standard libraies
#include <iostream>
using namespace std;

class hazard_detection
{
public:



private:

};


int main()
{
    cv::Mat img_src = cv::imread("book2.jpg");
    cv::Mat img_dst = cv::imread("book1.jpg");

    // convert the color into gray scale
    cv::cvtColor(img_src, img_src, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_dst, img_dst, cv::COLOR_BGR2GRAY);

    // Variables to store the key points and descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptor1, descriptor2;

    // Declare the SIFT
    cv::Ptr<cv::xfeatures2d::SIFT> feature = cv::xfeatures2d::SIFT::create();

    // Detect
    feature->detect(img_src, keypoints1);
    feature->detect(img_dst, keypoints2);

    // Compute
    feature->compute(img_src, keypoints1, descriptor1);
    feature->compute(img_dst, keypoints2, descriptor2);

    // Match
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher;
    matcher.match(descriptor1, descriptor2, matches);

    // Visualize
    cv::Mat img_matches;
    cv::drawMatches(img_src, keypoints1, img_dst, keypoints2, matches, img_matches);
    cv::imshow("img_matches", img_matches);

    // Push the points
    std::vector<cv::Point2f> pts_src, pts_dst;

    // Build the point set from the good matches features
    for(size_t i(0); i < matches.size(); ++i)
    {
        pts_src.push_back(keypoints1[matches[i].queryIdx].pt);
        pts_dst.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    
    // TODO(htsui): redo this from the self-implement RANSAC
    vector<uchar>inliersMask(pts_src.size());
    cv::Mat H = cv::findHomography(pts_src, pts_dst, cv::FM_RANSAC, 3.0, inliersMask, 1000);


    std::cout << "OPENCV RANSAC H = " << std::endl;
    std::cout << H << std::endl;
    cv::Mat img_out_RANSAC_H;
    cv::warpPerspective(img_src, img_out_RANSAC_H, H, img_src.size());
    cv::imshow("img_warp_RANSAC_H", img_out_RANSAC_H);



    // select 4 points from source image
    std::vector<cv::Point2f> pts_1;
    pts_1.push_back(cv::Point2f(141, 131));
    pts_1.push_back(cv::Point2f(480, 159));
    pts_1.push_back(cv::Point2f(493, 630));
    pts_1.push_back(cv::Point2f(64, 601));

    std::vector<cv::Point2f> pts_2;
    pts_2.push_back(cv::Point2f(318, 256));
    pts_2.push_back(cv::Point2f(534, 372));
    pts_2.push_back(cv::Point2f(316, 670));
    pts_2.push_back(cv::Point2f(73, 473));



    // Ransac
    // Under ransac you should know which problem are you solving
    // otherwise it will just be a coupled implemenation
    float p_success = 0.99;
    ransac::RANSAC rs(p_success);
    cv::Mat H_htsui;
    rs.compute(pts_src, pts_dst, H_htsui);
    cv::Mat img_htsui;
    cv::warpPerspective(img_src, img_htsui, H_htsui, img_src.size());
    cv::imshow("img_htsui", img_htsui);


    // OpenCV implementation
    cv::Mat h = cv::findHomography(pts_1, pts_2);
    std::cout << "opencv fix point h = " << std::endl;
    std::cout << h << std::endl;
    cv::Mat img_out_fix_h;
    cv::warpPerspective(img_src, img_out_fix_h, h, img_src.size());
    cv::imshow("img_warp_fix_h", img_out_fix_h);


    //cv::imshow("img_src", img_src);
    //cv::imshow("img_dst", img_dst);
    
    
    cv::waitKey(0);
}