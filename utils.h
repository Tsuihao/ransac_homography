#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>

// The following code is copied from Ref: https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/libmv_homography.cc#35

// A parameterization of the 2D homography matrix that uses 8 parameters so
// that the matrix is normalized (H(2,2) == 1).
// The homography matrix H is built from a list of 8 parameters (a, b,...g, h)
// as follows
//
//         |a b c|
//     H = |d e f|
//         |g h 1|
//
template<typename T = float>
class Homography2DNormalizedParameterization {
 public:
  typedef Eigen::Matrix<T, 8, 1> Parameters;     // a, b, ... g, h
  typedef Eigen::Matrix<T, 3, 3> Parameterized;  // H
  // Convert from the 8 parameters to a H matrix.
  static void To(const Parameters &p, Parameterized *h) {
    *h << p(0), p(1), p(2),
          p(3), p(4), p(5),
          p(6), p(7), 1.0;
  }
  // Convert from a H matrix to the 8 parameters.
  static void From(const Parameterized &h, Parameters *p) {
    *p << h(0, 0), h(0, 1), h(0, 2),
          h(1, 0), h(1, 1), h(1, 2),
          h(2, 0), h(2, 1);
  }
};

#endif