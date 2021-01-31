#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/mat.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;
using namespace Eigen;

int find_minEigenValue3(Vector3cd v) {
  double ev_min = 100000000000;
  int min_index = 3;
  for (int i=2; i>=0; i--) {
    if (ev_min > v[i].real()*v[i].real()) {
      ev_min = v[i].real()*v[i].real();
      min_index = i;
    }
  }
  return min_index;
}

Matrix3d calc_CrossMatrix(Vector3d v) {
  Matrix3d v_cross;
  v_cross << 0.0, -1*v[2], v[1],
             v[2], 0.0, -1*v[0],
             -1*v[1], v[0], 0.0;
  return v_cross;
}
