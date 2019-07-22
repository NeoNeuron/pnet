#include "common_header.h"
// Fast code for exp(x)
#include "fmath.hpp"
//#define exp(x) fmath::expd(x)

template<typename T>
inline T my_expd(const T &x)
{ return exp(x); }

template<>
inline double my_expd(const double &x)
{ return fmath::expd(x); }

#define exp(x) my_expd(x)

#include "io.h"
#include <chrono>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;

bool comp(const double &x, const double &y) {return x < y;}

int main(int argc, const char* argv[]) {
	string dir;
	auto start = chrono::system_clock::now();
	// config program options:
	po::options_description desc("All options");
	desc.add_options()
		("help,h", "produce help message")
		("prefix", po::value<string>(&dir)->default_value("./"), "prefix of working folder")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.count("help")) {
		cout << desc << '\n';
		return 1;
	}
	// import data
	vector<vector<double> > v;
	vector<vector<bool> > mat;
	Read2DBin(dir + "V.bin", v);
	Read2D(dir + "mat.csv", mat);
	vector<vector<double> > ras;
	Read2D(dir + "raster.csv", ras);
	
	// combining the inputing spiking events
	vector<vector<double> > spikes(mat.size());
  #pragma omp parallel for
	for (int i = 0; i < mat.size(); i ++) {
		for (int j = 0; j < mat.size(); j ++) {
			if (mat[i][j]) {
				spikes[i].insert(spikes[i].end(), ras[j].begin(), ras[j].end());
			}
		}
		sort(spikes[i].begin(), spikes[i].end(), comp);
	}
	
	// define neuronal paramters;
	double tau_Er = 1.0;	// (ms) rising const for exc conductance;
	double tau_Ed = 2.0;	// (ms) decay  const for exc conductance;
	double dt = 0.5;
	double ve = 14.0/3.0;
	double se = 5e-3;
	double exp_r = exp(-dt / tau_Er);
	double exp_d = exp(-dt / tau_Ed);
	double factor = (exp_d - exp_r)*tau_Ed*tau_Er/(tau_Ed - tau_Er);
	vector<double> t(v.size());
	t[0] = dt;
	for (int i = 1; i < v.size(); i ++) {
		t[i] = t[i-1] + dt;
	}
	vector<double>::iterator counter;
	vector<vector<double> > G(v.size(), vector<double>(v.begin()->size(), 0.0));
	vector<vector<double> > H(v.size(), vector<double>(v.begin()->size(), 0.0));
  #pragma omp parallel for private(counter)
	for (int i = 0; i < mat.size(); i ++) {
		counter = spikes[i].begin();
		for (int j = 0; j < t.size(); j ++) {
			if (j != 0) {
				G[j][i] = exp_d*G[j-1][i] + factor*H[j-1][i];
				H[j][i] = H[j-1][i]*exp_r;
			}
			if (counter != spikes[i].end()) {
				while (*counter < t[j]) {
					G[j][i] += se*(exp((*counter - t[j])/tau_Ed) - exp((*counter - t[j])/tau_Er))*tau_Ed*tau_Er/(tau_Ed - tau_Er);
					H[j][i] += se*exp((*counter - t[j])/tau_Er);
					counter ++;
					if (counter == spikes[i].end()) break;
				}
			}
		}
	}
  //#pragma omp parallel for
	//for (int i = 0; i < v.size(); i ++) {
	//	for (int j = 0; j < v[i].size(); j ++) {
	//		G[i][j] *= ve - v[i][j];
	//	}
	//}
	Print2DBin(dir + "GEc.bin", G, "trunc");

	auto finish = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = finish-start;
	printf(">> Processing : \t%3.3f s\n", elapsed_seconds.count());
	return 0;
}
