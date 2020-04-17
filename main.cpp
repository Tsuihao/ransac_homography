// standard libraies
#include <iostream>

// opencv 
#include <opencv2/opencv.hpp>          //warpPerspective
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp> // findHomography
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

// lib
#include "ransac.hpp"
using namespace std;

int main()
{
    cv::Mat img_src = cv::imread("../images/book2.jpg");
    cv::Mat img_dst = cv::imread("../images/book1.jpg");

    // convert the color into gray scale
    cv::cvtColor(img_src, img_src, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_dst, img_dst, cv::COLOR_BGR2GRAY);
    
    // ----------  Reference homography - Baseline Homography -----------------
    std::vector<cv::Point2f> pts_src_fix;
    pts_src_fix.push_back(cv::Point2f(141, 131));
    pts_src_fix.push_back(cv::Point2f(480, 159));
    pts_src_fix.push_back(cv::Point2f(493, 630));
    pts_src_fix.push_back(cv::Point2f(64, 601));

    std::vector<cv::Point2f> pts_dst_fix;
    pts_dst_fix.push_back(cv::Point2f(318, 256));
    pts_dst_fix.push_back(cv::Point2f(534, 372));
    pts_dst_fix.push_back(cv::Point2f(316, 670));
    pts_dst_fix.push_back(cv::Point2f(73, 473));

    // OpenCV implementation
    cv::Mat h = cv::findHomography(pts_src_fix, pts_dst_fix);
    std::cout << "opencv fix point h = " << std::endl;
    std::cout << h << std::endl;
    cv::Mat img_out_fix_h;
    cv::warpPerspective(img_src, img_out_fix_h, h, img_src.size());
    cv::imshow("img_baseline_homography", img_out_fix_h);

    // ------------------------- RANSAC ----------------------------------
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
    
    // OpenCV RANSAC
    vector<uchar>inliersMask(pts_src.size());
    cv::Mat H = cv::findHomography(pts_src, pts_dst, cv::FM_RANSAC, 3.0, inliersMask, 1000);
    std::cout << "OPENCV RANSAC H = " << std::endl;
    std::cout << H << std::endl;

    cv::Mat img_opencv_RANSAC;
    cv::warpPerspective(img_src, img_opencv_RANSAC, H, img_src.size());
    cv::imshow("img_OpenCV_RANSAC", img_opencv_RANSAC);

    
    // Comparing with Self-implemented Ransac
    float p_success = 0.99;
    ransac::RANSAC rs(p_success);
    cv::Mat H_htsui;
    rs.compute(pts_src, pts_dst, H_htsui);
    //rs.compute(pts_src_fix, pts_dst_fix, H_htsui);

    cv::Mat img_htsui_RANSAC;
    cv::warpPerspective(img_src, img_htsui_RANSAC, H_htsui, img_src.size());
    cv::imshow("img_htsui_RANSAC", img_htsui_RANSAC);

    // Comparing with nlsolver

    cv::Mat H_nlsolver_1000;
    Eigen::Matrix3f H_1000;
    H_1000 << 0.643212, -0.279826, 299.501,
         0.167107, 1.32003, 0,
         -0.000587184, 0.00114689, 1;

    cv::eigen2cv(H_1000, H_nlsolver_1000);

    cv::Mat img_nlsolver_1000;
    cv::warpPerspective(img_src, img_nlsolver_1000, H_nlsolver_1000, img_src.size());
    cv::imshow("img_nlsolver_1000", img_nlsolver_1000);


    cv::Mat H_nlsolver_100;
    Eigen::Matrix3f H_100;
    H_1000 << -4.16394, 5.69776, 361.468,
         -5.28821, 8.82406, 0,
         -0.0130028, 0.0176786, 1;

    cv::eigen2cv(H_100, H_nlsolver_100);

    cv::Mat img_nlsolver_100;
    cv::warpPerspective(img_src, img_nlsolver_100, H_nlsolver_100, img_src.size());
    cv::imshow("img_nlsolver_100", img_nlsolver_100);

    cv::waitKey(0);
}