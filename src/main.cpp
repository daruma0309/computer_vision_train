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
  int point_num = 100;
  sort(matches.begin(), matches.end(), [](DMatch a, DMatch b) {return a.distance < b.distance; });
  matches.resize(point_num);

  // 3次元復元に使うデータ
  vector<Point2f> points1_view, points2_view;
  for (int i=0; i<matches.size(); i++) {
    int idx_2 = matches[i].trainIdx;
    int idx_1 = matches[i].queryIdx;
    Point2f center(center_col, center_row);
    points1_view.push_back(keypoints1[idx_1].pt - center);
    points2_view.push_back(keypoints2[idx_2].pt - center);
  }

  Mat resImg;
  drawMatches(img_1, keypoints1, img_2, keypoints2, matches, resImg);
  cv::imshow("image", resImg);
	cv::waitKey();

////////////////////////////////////////////////////////////////////////////////
  // 3次元復元
  int point_num_uesd = 30;
  vector<Point2f> points1, points2;
  for (int i=0; i<point_num_uesd; i++) {
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
  double eps = 0.0000001; // ToDo
  Matrix<double, 9, 1> theta, theta0;

  // xiを設定
  vector<Matrix<double, 9, 1>> xis;
  xis.resize(points1.size());
  for (int i=0; i<points1.size(); i++) {
    Matrix<double, 9, 1> xi;
    xi << points1[i].x*points2[i].x,
          points1[i].x*points2[i].y,
          f0*points1[i].x,
          points1[i].y*points2[i].x,
          points1[i].y*points2[i].y,
          f0*points1[i].y,
          f0*points2[i].x,
          f0*points2[i].y,
          f0*f0;
    xis[i] = xi;
    //cout << "[DEBUG] xis[i] = " << xis[i] << endl;
  }

  // V0を設定
  vector<Matrix<double, 9, 9>> V0s;
  V0s.resize(points1.size());
  for (int i=0; i<points1.size(); i++) {
    Matrix<double, 9, 9> V0;
    V0 << points1[i].x*points1[i].x + points2[i].x*points2[i].x, points2[i].x*points2[i].y, f0*points2[i].x, points1[i].x*points1[i].y, 0, 0, f0*points1[i].x, 0, 0,
          points2[i].x*points2[i].y, points1[i].x*points1[i].x + points2[i].y*points2[i].y, f0*points2[i].y, 0, points1[i].x*points1[i].y, 0, 0, f0*points1[i].x, 0,
          f0*points2[i].x, f0*points2[i].y, f0*f0, 0, 0, 0, 0, 0, 0,
          points1[i].x*points1[i].y, 0, 0, points1[i].y*points1[i].y + points2[i].x*points2[i].x, points2[i].x*points2[i].y, f0*points2[i].x, f0*points1[i].y, 0, 0,
          0, points1[i].x*points1[i].y, 0, points2[i].x*points2[i].y, points1[i].y*points1[i].y + points2[i].y*points2[i].y, f0*points2[i].y, 0, f0*points1[i].y, 0,
          0, 0, 0, f0*points2[i].x, f0*points2[i].y, f0*f0, 0, 0, 0,
          f0*points1[i].x, 0, 0, f0*points1[i].y, 0, 0, f0*f0, 0, 0,
          0, f0*points1[i].x, 0, 0, f0*points1[i].y, 0, 0, f0*f0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0;
    V0s[i] = V0;
    cout << "[DUBUG] V0s[i] = " << V0s[i] << endl;
  }

  int loop_max = 50;
  //double ev0 = 100000000000;

  for (int i=0; i<loop_max; i++) {
    cout << "cnt = "<< i << endl;

    // M, L, Xを設定
    Matrix<double, 9, 9> M, L, X;
    set_zeroMatrix9d(M);
    set_zeroMatrix9d(L);
    set_zeroMatrix9d(X);
    for (int j=0; j<points1.size(); j++) {
      M = M + (W[j]*(xis[j]*xis[j].transpose()));
      L = L + (pow(W[j], 2.0)*pow(xis[j].dot(theta), 2.0)*V0s[j]);
    }
    M = M/points1.size();
    L = L/points1.size();
    X = M - L;
    //cout << "[DEBUG] M = " << M << endl;
    //cout << "[DEBUG] L = " << L << endl;
    //cout << "[DEBUG] X = " << X << endl;

    // 固有値分解
    EigenSolver<Matrix<double, 9, 9>> es(X);
    if (es.info() != Success) abort();
    cout << "[DEBUG] eigen values = \n"<< es.eigenvalues() << endl;
    VectorXcd evs = es.eigenvalues();

    // 最小固有値を探す
    double ev_min = 100000000000;
    int min_index = 8;
    for (int j=8; j>=0; j--) {
      if (ev_min > evs(j, 0).real()*evs(j, 0).real()) {
        ev_min = evs(j, 0).real()*evs(j, 0).real();
        min_index = j;
      }
    }

    // 最小固有値のベクトルをthetaに代入
    cout << "[DEBUG] eigen vectors = \n"<< es.eigenvectors() << endl;
    VectorXcd  min_eigen_vector = es.eigenvectors().col(min_index);
    for (int j=0; j<9; j++) {
      theta(j, 0) = min_eigen_vector(j, 0).real();
    }
    cout << "[DEBUG] theta = " << theta << endl;

    // W, theta0を更新
    for (int j=0; j<points1.size(); j++) {
      W[j] = 1.0 / theta.dot(V0s[j]*theta);
      //cout << "[DEBUG] W[j] = " << W[j] << endl;
    }

    // 最初のループでは
    if (i == 0) {
      theta0 = theta;
      continue;
    }

    // 終了条件
    cout << "[DEBUG] (theta-theta0).dot(theta-theta0) = " << (theta-theta0).dot(theta-theta0) << endl;
    if ((theta-theta0).dot(theta-theta0) < eps) break;

    theta0 = theta;
  }

  Matrix3d F;
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      F(i, j) = theta(i*3+j, 0);
    }
  }
  cout << "F = " << F << endl;

  //特異値分解．
  JacobiSVD<Matrix3d> SVD(F, ComputeFullU | ComputeFullV);
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

  // 基礎行列Fの定義式を計算
  for (int i=0; i<points1.size(); i++) {
    Vector3d x1, x2;
    x1 << points1[i].x/f0, points1[i].y/f0, 1.0;
    x2 << points2[i].x/f0, points2[i].y/f0, 1.0;
    //cout << "[DEBUG] xtFx = " << x2.transpose()*F_cor*x1 << endl;
  }

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
  eta = pow((F_cor.transpose()*k).norm(), 2.0) - k.dot(F_cor*F_cor.transpose()*F_cor*k) * pow((e1.cross(k)).norm(), 2.0) / k.dot(F_cor*k);
  xi = xi / (pow((e2.cross(k)).norm(), 2.0) * pow((F_cor.transpose()*k).norm(), 2.0) - pow(k.dot(F_cor*k), 2.0));
  eta = eta / (pow((e1.cross(k)).norm(), 2.0) * pow((F_cor*k).norm(), 2.0) - pow(k.dot(F_cor*k), 2.0));
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
    t(i, 0) = min_eigen_vector_EEt(i, 0).real();
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

  // カメラ行列
  Matrix3d f2f0;
  Matrix<double, 3, 4> P1, P2;
  P1 << f1, 0, 0, 0,
        0, f1, 0, 0,
        0, 0, f0, 0;
  f2f0 << f2, 0, 0,
          0, f2, 0,
          0, 0, f0;
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      Matrix3d Rt = R.transpose();
      P2(i, j) = Rt(i, j);
    }
    Vector3d Rt_t = -1*R.transpose()*t;
    P2(i, 3) = Rt_t(i, 0);
  }
  P2 = f2f0 * P2;
  cout << "P1 = " << P1 << endl;
  cout << "P2 = " << P2 << endl;

  // 三角測量
  vector<Vector3d> Xs;
  for (int i=0; i<points1_view.size(); i++) {
    Matrix<double, 4, 3> T;
    Vector4d p;
    T << f0*P1(0, 0)-points1_view[i].x*P1(2, 0), f0*P1(0, 1)-points1_view[i].x*P1(2, 1), f0*P1(0, 2)-points1_view[i].x*P1(2, 2),
         f0*P1(1, 0)-points1_view[i].y*P1(2, 0), f0*P1(1, 1)-points1_view[i].y*P1(2, 1), f0*P1(1, 2)-points1_view[i].y*P1(2, 2),
         f0*P2(0, 0)-points2_view[i].x*P2(2, 0), f0*P2(0, 1)-points2_view[i].x*P2(2, 1), f0*P2(0, 2)-points2_view[i].x*P2(2, 2),
         f0*P2(1, 0)-points2_view[i].y*P2(2, 0), f0*P2(1, 1)-points2_view[i].y*P2(2, 1), f0*P2(1, 2)-points2_view[i].y*P2(2, 2);
    p << f0*P1(0, 3)-points1_view[i].x*P1(2, 3),
         f0*P1(1, 3)-points1_view[i].y*P1(2, 3),
         f0*P2(0, 3)-points2_view[i].x*P2(2, 3),
         f0*P2(1, 3)-points2_view[i].y*P2(2, 3);

    Matrix3d TtT;
    Vector3d Ttp;
    TtT = T.transpose()*T;
    Ttp = -1*T.transpose()*p;

    // LU分解を使った連立方程式の解
    Vector3d X;
    FullPivLU<Matrix3d> lu(TtT);
    X = lu.solve(Ttp);
    Xs.push_back(X);
    //cout << "[DEBUG] X = " << X << endl;
  }

  string filename = "data.txt";
  ofstream outputfile;
  outputfile.open(filename, std::ios::out);
  for (int i=0; i<Xs.size(); i++) {
    outputfile << Xs[i](0, 0) << " " << Xs[i](1, 0) << " " <<  Xs[i](2, 0) << endl;
  }
  outputfile.close();
/*
  FILE *gp;
  gp = popen("gnuplot -persist","w");
  fprintf(gp, "set xlabel \"X axis\"\n");
  fprintf(gp, "set ylabel \"Z axis\"\n");
  fprintf(gp, "plot '-' with points pointsize 2\n");
  for (int i=0; i<Xs.size(); i++) {
    fprintf(gp,"%f\t%f\n", Xs[i](0), Xs[i](2));
  }
  pclose(gp);
*/
  cout << "End." << endl;
  return 0;
}
