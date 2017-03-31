#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_      = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0.0000,
              0.0000, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0.0000, 0.00,
              0.00, 0.0009, 0.00,
              0.00, 0.0000, 0.09;

  //measurement matrix - radar
  H_laser_ << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0;

  noise_ax_ = 9;
  noise_ay_ = 9;

  /** Declare vectors and matricies for Kalman Filter */
  ekf_.x_ = VectorXd(4);
  ekf_.P_ = MatrixXd(4,4);
  ekf_.F_ = MatrixXd(4,4);
  ekf_.Q_ = MatrixXd(4,4);

  //Create and Initialize State Transition Matrix
  ekf_.F_ << 1,0,0,0,
		     0,1,0,0,
		     0,0,1,0,
		     0,0,0,1;

  //Create an Initialize Process Noise Matrix
  ekf_.Q_ << 0,0,0,0,
		     0,0,0,0,
		     0,0,0,0,
		     0,0,0,0;

  //Create and Initialize Covariance Matrix
  ekf_.P_ << 1,0,   0,    0,
		       0,1,   0,    0,
		       0,0,1000,    0,
		       0,0,   0, 1000;


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 0, 0, 0, 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float range       = measurement_pack.raw_measurements_[0];
      float phi         = measurement_pack.raw_measurements_[1];
      float range_rate  = measurement_pack.raw_measurements_[2];
      ekf_.x_[0]        = range*cos(phi);
      ekf_.x_[1]        = range*sin(phi);
      ekf_.x_[2]        = range_rate*cos(phi);
      ekf_.x_[3]        = range_rate*sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      //In the case where px=py=0 initialize to px=py=1e-4 as suggested in project description
      if(measurement_pack.raw_measurements_[0] == 0 && measurement_pack.raw_measurements_[1] == 0){
          ekf_.x_[0] = 1e-4;
          ekf_.x_[1] = 1e-4;
      }
      else{
          ekf_.x_[0] = measurement_pack.raw_measurements_[1];
          ekf_.x_[1] = measurement_pack.raw_measurements_[1];
      }


      ekf_.x_[2] = 0;
      ekf_.x_[3] = 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  //Compute elapsed time since last epoch
  float dt = (measurement_pack.timestamp_ - previous_timestamp_)/1e6;
  previous_timestamp_ = measurement_pack.timestamp_;
  if(dt == 0){
	  cout << "/----------------------------/"<< endl;
	  std::cout << "Simultaneous Measurements\n";
	  dt = 1e-4;
  }

  // Update State Transition Matrix (F)
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  //Update Process Covariance Matrix (Q)
  float dt_2 =   dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4/4*noise_ax_,  0,                dt_3/2*noise_ax_, 0,
			 0,                 dt_4/4*noise_ay_, 0,                dt_3/2*noise_ay_,
			 dt_3/2*noise_ax_,  0,                dt_2*noise_ax_,   0,
			 0,                 dt_3/2*noise_ay_, 0,                dt_2*noise_ay_;

  //Time Update Equations
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  /** Use the sensor type to perform the update step. * Update the state and covariance matrices. */
  VectorXd innovation_z;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
	ekf_.R_ = R_radar_;
	innovation_z = measurement_pack.raw_measurements_ - GetRadarMeasurement();
  } else {
    // Laser updates
	ekf_.H_ = H_laser_;
	ekf_.R_ = R_laser_;
	innovation_z = measurement_pack.raw_measurements_ - GetLaserMeasurement();
  }

  ekf_.UpdateEKF(innovation_z);

  // print the output
  cout << "/----------------------------/"<< endl;
  cout << "Timestamp = " << measurement_pack.timestamp_ << endl;
  cout << "dt = " << dt << endl;
  cout << "x_ = " << endl;
  cout << ekf_.x_ << endl;
  cout << "P_ = " << endl;
  cout << ekf_.P_ << endl;
}

VectorXd FusionEKF::GetRadarMeasurement(){
	// Extract the Cartesian states
	float px, py, vx, vy, range, phi, range_rate;
	px = ekf_.x_[0];
	py = ekf_.x_[1];
	vx = ekf_.x_[2];
	vy = ekf_.x_[3];
	// Convert to polar coordinates
	range      = sqrt(px*px + py*py);
	// Protect against divide by zeroes
	if(fabs(px) > 1e-4){
		phi = atan2(py,px);
	}
	else{
		phi = 0.0;
	}
	if(fabs(range) > 1e-4){
		range_rate = (px*vx + py*vy)/range;
	}
	else{
		range_rate = 0.0;
	}
    //Fill radar measurement vector
	VectorXd y_radar(3);
	y_radar[0] = range;
	y_radar[1] = phi;
	y_radar[2] = range_rate;
	return y_radar;
}

VectorXd FusionEKF::GetLaserMeasurement(){
	VectorXd y_laser(4);
	y_laser = ekf_.H_ * ekf_.x_;
	return y_laser;
}
