#include "myUtil.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )	{
  string filename1 = "pic/pic1.jpg";
  string filename2 = "pic/pic3.jpg";

  Mat img_1_big = imread(filename1, 0);
  Mat img_2_big = imread(filename2, 0);

  Mat img_1, img_2;
  resize(img_1_big, img_1, Size(0,0), 0.15, 0.15);
  resize(img_2_big, img_2, Size(0,0), 0.15, 0.15);

  int rows = img_1.rows;
  int cols = img_1.cols;
  cout << rows << "," << cols << endl;

  int center_row = rows/2;
  int center_col = cols/2;
  double f0 = cols;
  cout << center_row << "," << center_col << endl;

  // 特徴点抽出, 特徴量計算
  Ptr<Feature2D> fdetector;
  fdetector = ORB::create();
  vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  fdetector->detectAndCompute(img_1, Mat(), keypoints1, descriptors1);
  fdetector->detectAndCompute(img_2, Mat(), keypoints2, descriptors2);

  // 対応点のマッチング
  Ptr<BFMatcher> bf;
  bf = BFMatcher::create(NORM_HAMMING, true);
  vector<DMatch> matches;
  bf->match(descriptors1, descriptors2, matches);

  cout << "match_num = " << matches.size() << endl;

  // マッチングが強い(距離が短い)ものから昇順にソート
  int point_num = 30;
  sort(matches.begin(), matches.end(), [](DMatch a, DMatch b) {return a.distance < b.distance; });
  matches.resize(point_num);

  for (int i=0; i<matches.size(); i++) {
    int idx_1 = matches[i].queryIdx;
    int idx_2 = matches[i].trainIdx;
    //cout << i << endl;
    //cout << static_cast<int>(keypoints1[idx_1].pt.x) << ", " << static_cast<int>(keypoints1[idx_1].pt.y) << "<-->";
    //cout << static_cast<int>(keypoints1[idx_2].pt.x) << ", " << static_cast<int>(keypoints1[idx_2].pt.y) << endl;
  }

  Mat resImg;
  drawMatches(img_1, keypoints1, img_2, keypoints2, matches, resImg);

  cv::imshow("image", resImg);
	cv::waitKey();

  vector<Point2f> points1, points2;
  for (int i=0; i<matches.size(); i++) {
    int idx_2 = matches[i].trainIdx;
    int idx_1 = matches[i].queryIdx;
    Point2f center(center_col, center_row);
    points1.push_back(keypoints1[idx_1].pt - center);
    points2.push_back(keypoints2[idx_2].pt - center);
    cout << points1[i].x << "," << points1[i].y << "<->";
    cout << points2[i].x << "," << points2[i].y << endl;
  }

  // OpenCVで基礎行列を計算
  //Mat F_cv = findFundamentalMat(points1, points2, CV_FM_8POINT);
  //cout << "F_vc = " << F_cv << endl;

  // FNS method
  vector<double> W(points1.size(), 1);
  Mat theta = Mat::zeros(9, 1, CV_64F);
  Mat theta0 = Mat::zeros(9, 1, CV_64F);
  double eps = 0.0001; // ToDo

  // xiを設定
  vector<Mat> xis;
  xis.resize(points1.size());
  for (int i=0; i<points1.size(); i++) {
    double xi_buf[9] = {
      points1[i].x*points2[i].x,
      points1[i].x*points2[i].y,
      f0*points1[i].x,
      points1[i].y*points2[i].x,
      points1[i].y*points2[i].y,
      f0*points1[i].y,
      f0*points2[i].x,
      f0*points2[i].y,
      f0*f0
    };
    Mat xi(9, 1, CV_64F, xi_buf);
    xis[i] = xi.clone();
  }

  // V0を設定
  vector<Mat> V0s;
  V0s.resize(points1.size());
  for (int i=0; i<points1.size(); i++) {
    double V0_buf[81] = {
      points1[i].x*points1[i].x + points2[i].x*points2[i].x, points2[i].x*points2[i].y, f0*points2[i].x, points1[i].x*points1[i].y, 0, 0, f0*points1[i].x, 0, 0,
      points2[i].x*points2[i].y, points1[i].x*points1[i].x + points2[i].y*points2[i].y, f0*points2[i].y, 0, points1[i].x*points1[i].y, 0, 0, f0*points1[i].x, 0,
      f0*points2[i].x, f0*points2[i].y, f0*f0, 0, 0, 0, 0, 0, 0,
      points1[i].x*points1[i].y, 0, 0, points1[i].y*points1[i].y + points2[i].x*points2[i].x, points2[i].x*points2[i].y, f0*points2[i].x, f0*points1[i].y, 0, 0,
      0, points1[i].x*points1[i].y, 0, points2[i].x*points2[i].y, points1[i].y*points1[i].y + points2[i].y*points2[i].y, f0*points2[i].y, 0, f0*points1[i].y, 0,
      0, 0, 0, f0*points2[i].x, f0*points2[i].y, f0*f0, 0, 0, 0,
      f0*points1[i].x, 0, 0, f0*points1[i].y, 0, 0, f0*f0, 0, 0,
      0, f0*points1[i].x, 0, 0, f0*points1[i].y, 0, 0, f0*f0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    Mat V0(9, 9, CV_64F, V0_buf);
    V0s[i] = V0.clone();
  }

  int cnt = 0;
  int loop_max = 50;
  double ev0 = 100000000000;

  for (int i=0; i<loop_max; i++) {
    cout << "cnt = "<< cnt << endl;

    // M, L, Xを設定
    Mat M = Mat::zeros(9, 9, CV_64F);
    Mat L = Mat::zeros(9, 9, CV_64F);
    Mat X = Mat::zeros(9, 9, CV_64F);
    for (int i=0; i<points1.size(); i++) {
      M = M + W[i]*(xis[i]*xis[i].t());
      Mat xiT_theta(xis[i].t()*theta);
      L = L + W[i]*W[i]*xiT_theta.data[0]*xiT_theta.data[0]*V0s[i];
    }
    M = M/points1.size();
    L = L/points1.size();
    X = M - L;

    // 固有値分解
    Matrix<double, 9, 9> eigen_X; // Eigen
    cv2eigen(X, eigen_X); // convert from cv::Mat to Eigen::Matrix
    EigenSolver<Matrix<double, 9, 9>> es(eigen_X);
    if (es.info() != Success) abort();
    //cout << "eigen values = \n"<< es.eigenvalues() << endl;
    VectorXcd evs = es.eigenvalues();

    // 最小固有値を探す
    double ev_min = 100000000000;
    int min_index = 8;
    for (int j=8; j>=0; j--) {
      if (ev_min > evs[j].real()*evs[j].real()) {
        ev_min = evs[j].real()*evs[j].real();
        min_index = j;
      }
    }

    // 最小固有値のベクトルをthetaに代入
    //cout << "eigen vectors = \n"<< es.eigenvectors() << endl;
    VectorXcd  min_eigen_vector = es.eigenvectors().col(min_index);
    for (int i=0; i<9; i++) {
      theta.at<double>(i,0) = min_eigen_vector[i].real();
    }

    // 終了条件
    if (norm(theta, theta0) < eps) break;
    else if (norm(theta, -1*theta0) < eps) break;
    if (ev0 - ev_min < 0) { // 理由は不明
      theta = theta0.clone();
      break;
    }
    ev0 = ev_min;

    //cout << "theta = " << theta << endl;

    // W, theta0を更新
    for (int i=0; i<points1.size(); i++) {
      Mat thetaT_V0_theta(theta.t() * (V0s[i] * theta));
      W[i] = 1.0 / thetaT_V0_theta.data[0];
    }
    theta0 = theta.clone();

    cnt++;
  }

  Mat F = theta.reshape(1, 3);
  cout << "F = " << F << endl;
/*
  // 基礎行列Fの定義式を計算
  for (int i=0; i<points1.size(); i++) {
    Mat x1 = (Mat_<double>(3,1) << points1[i].x/f0, points1[i].y/f0, 1.0);
    Mat x2 = (Mat_<double>(3,1) << points2[i].x/f0, points2[i].y/f0, 1.0);
    cout << x2.t()*F*x1 << endl;
  }
*/
  Matrix3d eigen_F; // Eigen
  cv2eigen(F, eigen_F); // convert from cv::Mat to Eigen::Matrix

  //特異値分解．
  JacobiSVD<Matrix3d> SVD(eigen_F, ComputeFullU | ComputeFullV);

  //特異値
  Vector3d sv = SVD.singularValues();
  //cout << "sigma = " << sv << endl;
  //行列UおよびV
  //cout << "U = " << SVD.matrixU() << endl;
  //cout << "V = " << SVD.matrixV() << endl;

  //もとの行列になるかを確認
  //Matrix3d F2;
  //F2 = SVD.matrixU() * SVD.singularValues().asDiagonal() * SVD.matrixV().transpose();
  //cout << "F2 = " << F2 << endl;

  // SDVによる事後補正
  Matrix3d F_cor;
  Matrix3d Sigma;
  Sigma << sv[0]/sqrt(sv[0]*sv[0] + sv[1]*sv[1]), 0, 0,
           0, sv[1]/sqrt(sv[0]*sv[0] + sv[1]*sv[1]), 0,
           0, 0, 0;
  F_cor = SVD.matrixU() * Sigma * SVD.matrixV().transpose();
  cout << "F_cor = " << F_cor << endl;

  // 焦点距離
  double f1, f2;
  EigenSolver<Matrix<double, 3, 3>> es_FFt(F_cor*F_cor.transpose());
  EigenSolver<Matrix<double, 3, 3>> es_FtF(F_cor.transpose()*F_cor);
  //cout << "eigen values = \n"<< es_FFt.eigenvalues() << endl;
  //cout << "eigen vectors = \n"<< es_FFt.eigenvectors() << endl;
  Vector3cd ev_FFt = es_FFt.eigenvalues();
  Vector3cd ev_FtF = es_FtF.eigenvalues();

  // 最小固有値のベクトルを計算
  int index_min_ev_FFt = find_minEigenValue3(ev_FFt);
  int index_min_ev_FtF = find_minEigenValue3(ev_FtF);
  Vector3cd  min_eigen_vector_FFt = es_FFt.eigenvectors().col(index_min_ev_FFt);
  Vector3cd  min_eigen_vector_FtF = es_FtF.eigenvectors().col(index_min_ev_FtF);
  Vector3d e1, e2, k;
  e1 << min_eigen_vector_FFt[0].real(),
        min_eigen_vector_FFt[1].real(),
        min_eigen_vector_FFt[2].real();
  e2 << min_eigen_vector_FtF[0].real(),
        min_eigen_vector_FtF[1].real(),
        min_eigen_vector_FtF[2].real();
  k << 0, 0, 1;

  //cout << "e1 = " << e1 << endl;
  //cout << "e2 = " << e2 << endl;

  double xi, eta;
  xi = pow((F_cor*k).norm(), 2.0) - (k.dot(F_cor*F_cor.transpose()*F_cor*k) * pow((e2.cross(k)).norm(), 2.0) / k.dot(F_cor*k));
  cout << "[DEBUG]" << xi << endl;
  eta = pow((F_cor.transpose()*k).norm(), 2.0) - k.dot(F_cor*F_cor.transpose()*F_cor*k) * pow((e1.cross(k)).norm(), 2.0) / k.dot(F_cor*k);
  xi = xi / (pow((e2.cross(k)).norm(), 2.0) * pow((F_cor.transpose()*k).norm(), 2.0) - pow(k.dot(F_cor*k), 2.0));
  eta = eta / (pow((e1.cross(k)).norm(), 2.0) * pow((F_cor*k).norm(), 2.0) - pow(k.dot(F_cor*k), 2.0));
  cout << "[DEBUG]" << xi << endl;
  f1 = f0 / sqrt(1.0 + xi);
  f2 = f0 / sqrt(1.0 + eta);

  cout << "f1 = " << f1 << endl;
  cout << "f2 = " << f2 << endl;

  // 並進パラメータ
  Matrix3d E; // 基本行列
  Matrix3d K1, K2;  // 内部パラメータ
  K1 << 1.0/f0, 0.0, 0.0,
        0.0, 1.0/f0, 0.0,
        0.0, 0.0, 1.0/f1;
  K2 << 1.0/f0, 0.0, 0.0,
        0.0, 1.0/f0, 0.0,
        0.0, 0.0, 1.0/f2;
  E = K1 * F_cor * K2;
  EigenSolver<Matrix<double, 3, 3>> es_EEt(E*E.transpose());
  //cout << "eigen values = \n"<< es_FFt.eigenvalues() << endl;
  //cout << "eigen vectors = \n"<< es_FFt.eigenvectors() << endl;
  Vector3cd ev_EEt = es_EEt.eigenvalues();
  // 最小固有値のベクトルを計算
  int index_min_ev_EEt = find_minEigenValue3(ev_EEt);
  Vector3cd  min_eigen_vector_EEt = es_EEt.eigenvectors().col(index_min_ev_EEt);
  //cout << "min_eigen_vector_EEt = " << min_eigen_vector_EEt << endl;
  Vector3d t;
  for (int i=0; i<3; i++) {
    t[i] = min_eigen_vector_EEt[i].real();
  }

  // tの符号チェック
  vector<Vector3d> x1s, x2s;
  for (int i=0; i<points1.size(); i++) {
    Vector3d x1, x2;
    x1 << points1[i].x/f1, points1[i].y/f1, 1.0;
    x2 << points2[i].x/f2, points2[i].y/f2, 1.0;
    x1s.push_back(x1);
    x2s.push_back(x2);
  }
  double t_check_sum = 0;
  for (int i=0; i<points1.size(); i++) {
    t_check_sum += t.dot(x1s[i].cross(E*x2s[i]));
  }
  if (t_check_sum <= 0.0) t = -1*t;
  cout << "t = " << t << endl;

  // 回転行列
  Matrix3d t_cross;
  t_cross = calc_CrossMatrix(t);
  Matrix3d t_cross_E;
  t_cross_E = -1*t_cross*E;
  //特異値分解
  JacobiSVD<Matrix3d> SVD2(t_cross_E, ComputeFullU | ComputeFullV);
  Matrix3d R, Lambda, UVt;
  UVt = SVD2.matrixU() * SVD2.matrixV().transpose();
  Lambda << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, UVt.determinant();
  R = SVD2.matrixU() * Lambda * SVD2.matrixV().transpose();
  cout << "R = " << R << endl;


  cout << "End." << endl;
  return 0;
}
