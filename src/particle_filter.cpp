/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 1000;
    weights.resize(num_particles);
    // gaussians for position noise
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // sample particles with initial positions from gaussians
    for (unsigned int i(0); i < num_particles; ++i)
    {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(1.0);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// gaussians for adding noise
	normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (vector<Particle>::size_type i(0); i < num_particles; ++i)
    {
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        // predict new positions
        if(fabs(yaw_rate) > 0.0001)
        {
            p_x = p_x + velocity / yaw_rate * ((sin(particles[i].theta + yaw_rate * delta_t)) - sin(particles[i].theta));
            p_y = p_y + velocity / yaw_rate * ((- cos(particles[i].theta + yaw_rate * delta_t)) + cos(particles[i].theta));
            p_theta = p_theta + yaw_rate * delta_t;
        }
        else
        {
            p_x = p_x + velocity*delta_t*cos(particles[i].theta);
            p_y = p_y + velocity*delta_t*sin(particles[i].theta);
        }
        particles[i].x = p_x + dist_x(gen);
        particles[i].y = p_y + dist_y(gen);
        particles[i].theta = p_theta + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (auto &observation : observations) {
        double min_dist = numeric_limits<double>::infinity();
        int land_id = 0;
        double obs_x = observation.x;
        double obs_y = observation.y;
        for (vector<LandmarkObs>::size_type j(0); j < predicted.size(); j++)
        {
            double pred_x = predicted[j].x;
            double pred_y = predicted[j].y;
            double distance = dist(obs_x, obs_y, pred_x, pred_y);
            if(distance < min_dist){
                min_dist = distance;
                land_id = j;
            }
        }
        observation.id = land_id;
    }
}

int ParticleFilter::findClosestLandmark(double obs_x, double obs_y, double p_x, double p_y, double sensor_range, const Map& map_landmarks)
{
    double min_dist = numeric_limits<double>::infinity();
    int min_i = -1;
    for (unsigned int i(0); i < map_landmarks.landmark_list.size(); ++i)
    {
        double land_x = map_landmarks.landmark_list[i].x_f;
        double land_y = map_landmarks.landmark_list[i].y_f;
        double distance_obs = dist(obs_x, obs_y, land_x, land_y);//sqrt((obs_x - land_x) * (obs_x - land_x) + (obs_y - land_y) * (obs_y - land_y));
        double distance_p = dist(p_x, p_y, land_x, land_y);
        if(distance_obs < min_dist && distance_p <= sensor_range){
            min_dist = distance_obs;
            min_i = i;
        }
    }
    return (min_i);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    weights.clear();
    for (vector<Particle>::size_type i(0); i < num_particles; ++i)
    {
        particles[i].weight = 1.0;
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;
        for (const auto &observation : observations) {
            // take only observations in sensor range
            // convert observations into map coordinates
            double map_obs_x = p_x + (cos(p_theta) * observation.x) - (sin(p_theta) * observation.y);
            double map_obs_y = p_y + (sin(p_theta) * observation.x) + (cos(p_theta) * observation.y);

            // find closest landmark
            int closest_landmark = findClosestLandmark(map_obs_x, map_obs_y, p_x, p_y, sensor_range, map_landmarks);
            double land_x = map_landmarks.landmark_list[closest_landmark].x_f;
            double land_y = map_landmarks.landmark_list[closest_landmark].y_f;

            // calculate multivariable probability
            double prob = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) *
                          exp((-1.0) * (((map_obs_x - land_x) * (map_obs_x - land_x)) / (2.0 * std_landmark[0] * std_landmark[0]) +
                                        ((map_obs_y - land_y) * (map_obs_y - land_y)) / (2.0 * std_landmark[1] * std_landmark[1])));
            particles[i].weight *= prob;

        }
        weights.push_back(particles[i].weight);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    std::vector<Particle> new_particles;
    default_random_engine gen;
    for (unsigned i = 0; i < num_particles; i++) {
        auto ind = d(gen);
        new_particles.push_back(std::move(particles[ind]));
    }
    particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
