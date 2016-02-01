// zener.cpp
// Anisotropic coarsening algorithms for 2D and 3D Monte Carlo methods
// Questions/comments to gruberja@gmail.com (Jason Gruber)

#ifndef ZENER_UPDATE
#define ZENER_UPDATE
#include"MMSP.hpp"
#include"anisotropy.hpp"
#include"zener.hpp"
#include<cmath>

namespace MMSP{

void generate(int dim, const char* filename)
{
	if (dim==1) {
		grid<1,int> initGrid(0,0,128);

		for (int i=0; i<nodes(initGrid); i++) {
			vector<int> x = position(initGrid,i);
			double d = 64.0-x[0];
			if (d<32.0) initGrid(i) = 2;
			else initGrid(i) = 1;
		}

		for (int j=0; j<50; j++) {
			int i = rand()%nodes(initGrid);
			vector<int> p = position(initGrid,i);
			for (int x=p[0]-1; x<=p[0]+1; x++)
				initGrid[x] = 0;
		}

		output(initGrid,filename);
	}

	if (dim==2) {
		grid<2,int> initGrid(0,0,128,0,128);

		for (int i=0; i<nodes(initGrid); i++) {
			vector<int> x = position(initGrid,i);
			double d = sqrt(pow(64.0-x[0],2)+pow(64.0-x[1],2));
			if (d<32.0) initGrid(i) = 2;
			else initGrid(i) = 1;
		}

		for (int j=0; j<50; j++) {
			int i = rand()%nodes(initGrid);
			vector<int> p = position(initGrid,i);
			for (int x=p[0]-1; x<=p[0]+1; x++)
				for (int y=p[1]-1; y<=p[1]+1; y++)
					initGrid[x][y] = 0;
		}

		output(initGrid,filename);
	}

	if (dim==3) {
		grid<3,int> initGrid(0,0,64,0,64,0,64);

		for (int i=0; i<nodes(initGrid); i++) {
			vector<int> x = position(initGrid,i);
			double d = sqrt(pow(32.0-x[0],2)+pow(32.0-x[1],2)+pow(32.0-x[2],2));
			if (d<16.0) initGrid(i) = 2;
			else initGrid(i) = 1;
		}

		for (int j=0; j<50; j++) {
			int i = rand()%nodes(initGrid);
			vector<int> p = position(initGrid,i);
			for (int x=p[0]-1; x<=p[0]+1; x++)
				for (int y=p[1]-1; y<=p[1]+1; y++)
					for (int z=p[2]-1; z<=p[2]+1; z++)
						initGrid[x][y][z] = 0;
		}

		output(initGrid,filename);
	}
}

void update(grid<1,int>& mcGrid, int steps)
{
	const double kT = 0.50;
	int gss = int(nodes(mcGrid));

	for (int step=0; step<steps; step++) {
		for (int h=0; h<nodes(mcGrid); h++) {
			// choose a random node
			int p = rand()%nodes(mcGrid);
			vector<int> x = position(mcGrid,p);
			int spin1 = mcGrid(p);

			if (spin1!=0) {
				// determine neighboring spins
				sparse<bool> neighbors;
				for (int i=-1; i<=1; i++) {
						int spin = mcGrid[x[0]+i];
						set(neighbors,spin) = true;
					}

				// choose a random neighbor spin
				int spin2 = index(neighbors,rand()%length(neighbors));

				if (spin1!=spin2 and spin2!=0) {
					// compute energy change
					double dE = -energy(spin1,spin2);
					for (int i=-1; i<=1; i++) {
							int spin = mcGrid[x[0]+i];
							dE += energy(spin,spin2)-energy(spin,spin1);
						}

					// compute boundary energy, mobility
					double E = energy(spin1,spin2);
					double M = mobility(spin1,spin2);

					// attempt a spin flip
					double r = double(rand())/double(RAND_MAX);
					if (dE<=0.0 and r<M*E) mcGrid(p) = spin2;
					if (dE>0.0 and r<M*E*exp(-dE/(E*kT))) mcGrid(p) = spin2;
				}
			}
			if (h%gss==0) ghostswap(mcGrid);
		}
	}
}

void update(grid<2,int>& mcGrid, int steps)
{
	const double kT = 0.50;
	int gss = int(sqrt(nodes(mcGrid)));

	for (int step=0; step<steps; step++) {
		for (int h=0; h<nodes(mcGrid); h++) {
			// choose a random node
			int p = rand()%nodes(mcGrid);
			vector<int> x = position(mcGrid,p);
			int spin1 = mcGrid(p);

			if (spin1!=0) {
				// determine neighboring spins
				sparse<bool> neighbors;
				for (int i=-1; i<=1; i++)
					for (int j=-1; j<=1; j++) {
						int spin = mcGrid[x[0]+i][x[1]+j];
						set(neighbors,spin) = true;
					}

				// choose a random neighbor spin
				int spin2 = index(neighbors,rand()%length(neighbors));

				if (spin1!=spin2 and spin2!=0) {
					// compute energy change
					double dE = -energy(spin1,spin2);
					for (int i=-1; i<=1; i++)
						for (int j=-1; j<=1; j++){
							int spin = mcGrid[x[0]+i][x[1]+j];
							dE += energy(spin,spin2)-energy(spin,spin1);
						}

					// compute boundary energy, mobility
					double E = energy(spin1,spin2);
					double M = mobility(spin1,spin2);

					// attempt a spin flip
					double r = double(rand())/double(RAND_MAX);
					if (dE<=0.0 and r<M*E) mcGrid(p) = spin2;
					if (dE>0.0 and r<M*E*exp(-dE/(E*kT))) mcGrid(p) = spin2;
				}
			}
			if (h%gss==0) ghostswap(mcGrid);
		}
	}
}

void update(grid<3,int>& mcGrid, int steps)
{
	const double kT = 0.75;
	int gss = int(sqrt(nodes(mcGrid)));

	for (int step=0; step<steps; step++) {
		for (int h=0; h<nodes(mcGrid); h++) {
			// choose a random node
			int p = rand()%nodes(mcGrid);
			vector<int> x = position(mcGrid,p);
			int spin1 = mcGrid(p);

			if (spin1!=0) {
				// determine neighboring spins
				sparse<bool> neighbors;
				for (int i=-1; i<=1; i++)
					for (int j=-1; j<=1; j++)
						for (int k=-1; k<=1; k++) {
							int spin = mcGrid[x[0]+i][x[1]+j][x[2]+k];
							set(neighbors,spin) = true;
						}

				// choose a random neighbor spin
				int spin2 = index(neighbors,rand()%length(neighbors));

				if (spin1!=spin2 and spin2!=0) {
					// compute energy change
					double dE = -energy(spin1,spin2);
					for (int i=-1; i<=1; i++)
						for (int j=-1; j<=1; j++)
							for (int k=-1; k<=1; k++) {
								int spin = mcGrid[x[0]+i][x[1]+j][x[2]+k];
								dE += energy(spin,spin2)-energy(spin,spin1);
							}

					// compute boundary energy, mobility
					double E = energy(spin1,spin2);
					double M = mobility(spin1,spin2);

					// attempt a spin flip
					double r = double(rand())/double(RAND_MAX);
					if (dE<=0.0 and r<M*E) mcGrid(p) = spin2;
					if (dE>0.0 and r<M*E*exp(-dE/(E*kT))) mcGrid(p) = spin2;
				}
			}
			if (h%gss==0) ghostswap(mcGrid);
		}
	}
}

} // namespace MC

#endif

#include"MMSP.main.hpp"
