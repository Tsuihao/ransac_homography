#ifndef RANSAC_H
#define RANSAC_H

// std
#include <math.h> // ceil, isnan
#include <limits> // numeric_limits
#include <random>
#include <algorithm>
#include <iterator>
#include <cassert> // assert

// Eigen
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/SVD>  // SVD

// opencv
#include <opencv2/core/eigen.hpp> // eigen2cv
#include <opencv2/core/core.hpp>

#include "utils.h"

#define VERBOSE true
#define DEBUG false

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

///////////////////////////////////////////////////////////////////////////////
// main compute function
void compute(const std::vector<cv::Point2f>& pts_src,
             const std::vector<cv::Point2f>& pts_dst, 
             cv::Mat& H)
{
    // cache the data set size
    setDatasetSize(pts_src, pts_dst);

    //update the iternation
    updateInterNum();

    // get sample points
    genRandomIndices(m_iter);

   
    for(size_t i(0); i < m_indicesList.size(); ++i)
    {   
         // declare containers
        std::vector<cv::Point2f> pts_out_src;
        std::vector<cv::Point2f> pts_out_dst;
    
        // get 4 points
        for(size_t k(0); k < m_minSet; ++k)
        {
            size_t index = m_indicesList[i][k];
            cv::Point2f src = pts_src[index];
            cv::Point2f dst = pts_dst[index];

            pts_out_src.push_back(src);
            pts_out_dst.push_back(dst);
        }

        // compute homography, use Eigen for the internal interface
        Eigen::Matrix3f homography;
        calHomographyFromLinerConstraint(pts_out_src, pts_out_dst, homography);
        //calHomographyFromSVD(pts_out_src, pts_out_dst, homography);

        // 3 x N matrix (N = datasetSize)
        MatrixXf eigen_pts_src(3, m_datasetSize);
        MatrixXf eigen_pts_dst(3, m_datasetSize);
        
        // take all points in & extend to Homogenous Coordinate
        buildUpEigenMatrixFromCvPoint2f(pts_src, pts_dst, eigen_pts_src, eigen_pts_dst);

        // compute the costFunction and update the liners
        size_t inliers = findInliers(eigen_pts_dst, eigen_pts_src, homography);

        // update BestModel
        if(inliers > m_maxInliers)
        {

            m_maxInliers = inliers;
            m_bestHomography = homography;

            std::cout << " ********************* Best model update ********************* " << std::endl;
            std::cout << "best inliers = " << m_maxInliers << std::endl;
            std::cout << "best H = " << std::endl;
            std::cout << m_bestHomography << std::endl;
        }
   
        
    }
    cv::eigen2cv(m_bestHomography, H);

}
///////////////////////////////////////////////////////////////////////////////
protected:
void setDatasetSize(const std::vector<cv::Point2f>& pts_src, const std::vector<cv::Point2f>& pts_dst)
{
    assert(pts_src.size() == pts_dst.size());
    m_datasetSize = pts_src.size();
}

///////////////////////////////////////////////////////////////////////////////
void buildUpEigenMatrixFromCvPoint2f(const std::vector<cv::Point2f>& pts_src,
                                     const std::vector<cv::Point2f>& pts_dst,
                                     MatrixXf& eigen_pts_src,
                                     MatrixXf& eigen_pts_dst)
{
    assert(pts_src.size() == pts_dst.size());

    // init vector with dataset size
    RowVectorXf v_pts_src_x(m_datasetSize), v_pts_src_y(m_datasetSize);
    RowVectorXf v_pts_dst_x(m_datasetSize), v_pts_dst_y(m_datasetSize);
    RowVectorXf v_ones(m_datasetSize); // for homogenous coordinate

    // fill in the eigen matrix 
    for(size_t i(0); i < pts_src.size(); ++i)
    {
        v_pts_src_x(0, i) = pts_src[i].x;
        v_pts_src_y(0, i) = pts_src[i].y;
        
        v_pts_dst_x(0, i) = pts_dst[i].x;
        v_pts_dst_y(0, i) = pts_dst[i].y;

        v_ones(0, i) = 1.0;  
    }

    // extend to homogenous coordinate
    /*
        | x1, x2, x3, ...................|
        | y1, y2, y3, ...................|
        |  1,  1,  1, ...................|
    */

    eigen_pts_src.row(0) << v_pts_src_x;
    eigen_pts_src.row(1) << v_pts_src_y;
    eigen_pts_src.row(2) << v_ones;

    eigen_pts_dst.row(0) << v_pts_dst_x;
    eigen_pts_dst.row(1) << v_pts_dst_y;
    eigen_pts_dst.row(2) << v_ones;

}
///////////////////////////////////////////////////////////////////////////////
// calculate the iteration number
void updateInterNum()
{
    /*
        N = log(1-p) / log(1-(1-e)^s)

        test 
        s = 2, e = 0.05 -> N = 2
        s = 2, e = 0.5  -> N = 17
        s = 4, e = 0.05 -> N = 3
        s = 4, e = 0.5  -> N = 72
        s = 8, e = 0.05 -> N = 5
        s = 8, e = 0.5  -> N = 1177
    */

    // adaptive prodecure for outliers ratio
    //m_iter = std::numeric_limits<int>::max(); // N start with infinity
    //int sampleCount = 0;

    //while(m_iter > sampleCount)
    //{
    //    m_iter = ceil(log(1 - m_p_success) / log(1 - pow((1 - m_ratioOutliers), m_minSet) ));
    //}
    if(VERBOSE) std::cout << "set iteration number N = " << m_iter << std::endl;
}
///////////////////////////////////////////////////////////////////////////////
// forward projection
// p' = H * p
// Here can do vectorization
std::vector<float> forwardProjectionSqaureRootError(const Eigen::MatrixXf&  pts_dst, const Eigen::MatrixXf& pts_src, const Eigen::Matrix3f& H)
{
    std::vector<float> error;
    error.reserve(pts_src.size());
    
    MatrixXf p_, p_diff;
    
    p_ = H * pts_src;
    
    // Every column needs to convert from "homogenous -> cartesian"
    for(size_t col(0); col < p_.cols(); ++col)
    {
        p_(0, col) /= p_(2, col);  // x
        p_(1, col) /= p_(2, col);  // y
    }
    
    // compute the diff
    p_diff = pts_dst - p_;

    // if(DEBUG)
    // {
    //     std::cout << "pts_src = " << pts_src << std::endl;
    //     std::cout << "p_ = " << p_ << std::endl;
    //     std::cout << "p_diff" << p_diff << std::endl;
    // }

    for(size_t col(0); col < p_diff.cols(); ++col)
    {
        float x_diff = p_diff(0, col);
        float y_diff = p_diff(1, col);
        float tmp = std::sqrt(x_diff*x_diff + y_diff*y_diff);
        error.push_back(tmp);
    }
    if(DEBUG)
    {
        std::cout << "forward projection SRE = " << std::endl;
        for (size_t i(0); i < error.size(); ++i)
        {
            std::cout << error[i] << "\t";
        }
        std::cout << std::endl;
    }
    return error;
}
///////////////////////////////////////////////////////////////////////////////
// backward projection
// p = H_inv * p'
std::vector<float>  backwardProjectionSqaureRootError(const Eigen::MatrixXf&  pts_dst, const Eigen::MatrixXf& pts_src, const Eigen::Matrix3f& H)
{
    std::vector<float> error;
    error.reserve(pts_src.size());

    MatrixXf p, p_diff;

    p = H.inverse() * pts_dst;
    
    // Every column needs to convert from "homogenous -> cartesian"
    for(size_t col(0); col < p.cols(); ++col)
    {
        p(0, col) /= p(2, col);  // x
        p(1, col) /= p(2, col);  // y
    }
    
    p_diff = pts_src - p;

    for(size_t col(0); col < p_diff.cols(); ++col)
    {
        float x_diff = p_diff(0, col);
        float y_diff = p_diff(1, col);
        float tmp = std::sqrt(x_diff*x_diff + y_diff*y_diff);
        error.push_back(tmp);
    }
    
    if(DEBUG)
    {
        std::cout << "backward projection SRE = " << std::endl;
        for (size_t i(0); i < error.size(); ++i)
        {
            std::cout << error[i] << "\t";
        }
        std::cout << std::endl;
    }

    return error;

}
///////////////////////////////////////////////////////////////////////////////
size_t findInliers(const Eigen::MatrixXf&  pts_dst, const Eigen::MatrixXf& pts_src, const Eigen::Matrix3f& H)
{
    size_t inliers = 0;
    
    // compute the cost function
    std::vector<float> forward_error_list = forwardProjectionSqaureRootError(pts_dst, pts_src, H);
    std::vector<float> backward_error_list = backwardProjectionSqaureRootError(pts_dst, pts_src, H);

    for(size_t i(0); i < backward_error_list.size(); ++i)
    {   
        float total_error = forward_error_list[i] + backward_error_list[i];
        if(DEBUG) std::cout << "total SRE =" << total_error << std::endl;
        if(total_error < m_thres)
        {
            inliers++;
        }
    }

    return inliers;
}
///////////////////////////////////////////////////////////////////////////////
void genRandomIndices(size_t count)
{   
    std::uniform_int_distribution<size_t> indexDistribution(0, m_datasetSize - 1);

    // reset
    m_indicesList.clear();
  
    for(size_t k(0); k < count; ++k)
    {
        cv::Vec4i tmp;
        
        // Assume we will always get 4 elements
        for(size_t i(0); i < m_minSet; ++i)
        {   
            bool valid = false;
            while(!valid)
            {
                tmp[i] = indexDistribution(m_rnd);
                valid = true;
                for(size_t j(0); j < i; ++j)
                {
                    if(tmp[j] == tmp[i])
                    {
                        valid = false;
                        break;
                    }
                }        
            }
        }

        // TODO(htsui): Check points
        // no 3 points are not on the same line
        m_indicesList.push_back(tmp);
    }
    if(VERBOSE)
    {
        std::cout << "m_indicesList size = " << m_indicesList.size() << std::endl;
    }
}
///////////////////////////////////////////////////////////////////////////////
void calHomographyFromSVD(const point2f_set& pts_src, 
                          const point2f_set& pts_dst, 
                          Eigen::Matrix3f& H)
{  
    // A * h = 0
    // A =8x9 matrix, h = 9x1 matrix, 
    // check slide 35: http://www.cse.psu.edu/~rtc12/CSE486/lecture16.pdf

    /*
        | x, y, 1, 0, 0, 0, -x'x, -x'y -x'|      |h11|     | 0 |
        | 0, 0, 0, x, y, 1, -y'x, -y'y -y'|      |h12|     | 0 |
        | ............................... |      |h13|     | 0 |
        | ............................... |      |h21|     | 0 |
        | ............................... |   *  |h22|  =  | 0 |
        | ............................... |      |h23|     | 0 |
        | ............................... |      |h31|     | 0 |
        | ............................... |      |h32|     | 0 |
        | ............................... |      |h33|     | 0 |
                                           8x9        9x1        8x1    
    */
    
    // symbol: x' replace with x_
    assert(pts_src.size() == pts_dst.size());
    assert(pts_src.size() == 4);

    typedef Eigen::Matrix<float, 8, 9> Mat8x9;
    typedef Eigen::Matrix<float, 9, 1> Vec9;

    Mat8x9 A;
    Vec9   h;

    for (size_t i(0); i < pts_src.size(); ++i)
    {
        float x  = pts_src[i].x;
        float y  = pts_src[i].y;
        float x_ = pts_dst[i].x;
        float y_ = pts_dst[i].y;

        // Fill the matrix A
        A(2*i, 0) = x;
        A(2*i, 1) = y;
        A(2*i, 2) = 1;
        A(2*i, 3) = 0;
        A(2*i, 4) = 0;
        A(2*i, 5) = 0;
        A(2*i, 6) = -x_ * x;
        A(2*i, 7) = -x_ * y;
        A(2*i, 8) = -x_;

        A(2*i+1, 0) = 0;
        A(2*i+1, 1) = 0;
        A(2*i+1, 2) = 0;
        A(2*i+1, 3) = x;
        A(2*i+1, 4) = y;
        A(2*i+1, 5) = 1;
        A(2*i+1, 6) = -y_ * x;
        A(2*i+1, 7) = -y_ * y;
        A(2*i+1, 8) = -y_;
    }
    
    // SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV ); 

    std::cout << "singular value = " << svd.singularValues() << std::endl;
    std::cout << "U = \n" << svd.matrixU() << std::endl;
    std::cout << "V = \n" << svd.matrixV() << std::endl;
    
    h = svd.matrixU().col(7);

    // vector form to Matrix
    Homography2DNormalizedParameterization<float>::To(h, &H);

    if(DEBUG)
    {
        std::cout << "A = " << std::endl;
        std::cout << A << std::endl;

        std::cout << "-------" << std::endl;
        std::cout << "Ah=0, H =" << std::endl;
        std::cout << H << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
void calHomographyFromLinerConstraint(const point2f_set& pts_src,
                                      const point2f_set& pts_dst,
                                      Eigen::Matrix3f& H)
{
    // A * h = b
    // A = 8x8 matrix, h = 8x1 matrix, b = 8x1
    // check slide 27: http://www.cse.psu.edu/~rtc12/CSE486/lecture16.pdf

    /*
        | x, y, 1, 0, 0, 0, -x'x, -x'y |      |h11|     | x' |
        | 0, 0, 0, x, y, 1, -y'x, -y'y |      |h12|     | y' |
        | ............................ |      |h13|     | .. |
        | ............................ |      |h21|     | .. |
        | ............................ |   *  |h22|  =  | .. |
        | ............................ |      |h23|     | .. |
        | ............................ |      |h31|     | .. |
        | ............................ |      |h32|     | .. |
                                        8x8        8x1        8x1    
    */

    // symbol: x' replace with x_
    assert(pts_src.size() == pts_dst.size());
    assert(pts_src.size() == 4);

    typedef Eigen::Matrix<float, 8, 8> Mat8;
    typedef Eigen::Matrix<float, 8, 1> Vec8;

    Mat8 A;
    Vec8 h, b;

    for (size_t i(0); i < pts_src.size(); ++i)
    {
        float x  = pts_src[i].x;
        float y  = pts_src[i].y;
        float x_ = pts_dst[i].x;
        float y_ = pts_dst[i].y;

        // Fill the matrix A
        A(2*i, 0) = x;
        A(2*i, 1) = y;
        A(2*i, 2) = 1;
        A(2*i, 3) = 0;
        A(2*i, 4) = 0;
        A(2*i, 5) = 0;
        A(2*i, 6) = -x_ * x;
        A(2*i, 7) = -x_ * y;

        A(2*i+1, 0) = 0;
        A(2*i+1, 1) = 0;
        A(2*i+1, 2) = 0;
        A(2*i+1, 3) = x;
        A(2*i+1, 4) = y;
        A(2*i+1, 5) = 1;
        A(2*i+1, 6) = -y_ * x;
        A(2*i+1, 7) = -y_ * y;

        // Fill the matrix b
        b(2*i,   0) = x_;
        b(2*i+1, 0) = y_;
    }

    h = A.fullPivLu().solve(b);
    
    // vector form to Matrix
    Homography2DNormalizedParameterization<float>::To(h, &H);

    if(DEBUG)
    {
        std::cout << "A = " << std::endl;
        std::cout << A << std::endl;
        std::cout << "b = " << std::endl;
        std::cout << b << std::endl;

        std::cout << "-------" << std::endl;
        std::cout << "Ah=b, H =" << std::endl;
        std::cout << H << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
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