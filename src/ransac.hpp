#ifndef RANSAC_HPP
#define RANSAC_HPP

// std
#include <math.h> // ceil, isnan
#include <limits> // numeric_limits
#include <random>
#include <algorithm>
#include <iterator>
#include <cassert> // assert

// Eigen
#include <Eigen/Core>
#include <Eigen/LU>   // Lx = b
#include <Eigen/SVD>  // SVD

// opencv
#include <opencv2/core/eigen.hpp> // eigen2cv
#include <opencv2/core/core.hpp>

#include "utils.hpp"

using point2f = cv::Point2f;
using point2f_set = std::vector<cv::Point2f>;

namespace ransac
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixXf;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> RowVectorXf;

class RANSAC
{
public:
    // ctor
    RANSAC(float prob_success) : 
        m_p_success(prob_success),
        m_ratioOutliers(1.0), // outliers ratio begin with 100%
        m_minSet(4),
        m_thres(10),         // TBD(htsui) forward and backward SSD error tolerance
        m_iter(2000),
        m_maxInliers(0)       // init maxInliner with 0
        {
            // Make sure the indicesList is the same size as iteration number
            m_indicesList.reserve(m_iter);
            m_rnd.seed(168);
        }

    // main compute function
    void compute(const std::vector<cv::Point2f>& pts_src,
                const std::vector<cv::Point2f>& pts_dst, 
                cv::Mat& H);

protected:

    void buildUpEigenMatrixFromCvPoint2f(const std::vector<cv::Point2f>& pts_src,
                                        const std::vector<cv::Point2f>& pts_dst,
                                        MatrixXf& eigen_pts_src,
                                        MatrixXf& eigen_pts_dst);

    // calculate the iteration number
    void updateInterNum();

    // forward projection
    // p' = H * p
    // Here can do vectorization
    std::vector<float> forwardProjectionSqaureRootError(const Eigen::MatrixXf&  pts_dst,
                                                        const Eigen::MatrixXf& pts_src,
                                                        const Eigen::Matrix3f& H);

    // backward projection
    // p = H_inv * p'
    std::vector<float>  backwardProjectionSqaureRootError(const Eigen::MatrixXf&  pts_dst,
                                                        const Eigen::MatrixXf& pts_src, 
                                                        const Eigen::Matrix3f& H);

    size_t findInliers(const Eigen::MatrixXf&  pts_dst, 
                    const Eigen::MatrixXf& pts_src, 
                    const Eigen::Matrix3f& H);

    void genRandomIndices(size_t count);


    void calHomographyFromSVD(const point2f_set& pts_src, 
                            const point2f_set& pts_dst, 
                            Eigen::Matrix3f& H);


    void calHomographyFromLinerConstraint(const point2f_set& pts_src,
                                        const point2f_set& pts_dst,
                                        Eigen::Matrix3f& H);

    void setDatasetSize(const std::vector<cv::Point2f>& pts_src, 
                        const std::vector<cv::Point2f>& pts_dst)
    {
        assert(pts_src.size() == pts_dst.size());
        m_datasetSize = pts_src.size();
    }

private:

    int m_minSet;                  // for homography is 4
    long int m_thres;              // threshold
    int m_iter;                    // interation
    float m_p_success;             // probility of sucess
    float m_ratioOutliers;         // ratio of outliers
    size_t m_datasetSize;          // totoal number of points
    size_t m_maxInliers;           // number of inliers
    Eigen::Matrix3f m_bestHomography;    // base on the inliner count
    std::vector<cv::Vec4i> m_indicesList;
    std::mt19937 m_rnd;
    std::vector<cv::Point2f> m_pts_src;
    std::vector<cv::Point2f> m_pts_dst;

}; // class RANSAC
} // namespace ransac
#endif