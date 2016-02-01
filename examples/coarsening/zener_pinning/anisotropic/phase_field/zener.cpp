// zener.cpp
// Anisotropic coarsening algorithms for 2D and 3D phase field methods
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
		grid<1,vector<double> > initGrid(3,0,128);

		for (int i=0; i<nodes(initGrid); i++) {
			vector<int> x = position(initGrid,i);
			initGrid(i)[0] = 0.0;
			initGrid(i)[1] = 0.0;
			double d = 64.0-x[0];
			if (d<32.0) initGrid(i)[2] = 1.0;
			else initGrid(i)[1] = 1.0;
		}

		for (int j=0; j<50; j++) {
			int i = rand()%nodes(initGrid);
			vector<int> p = position(initGrid,i);
			for (int x=p[0]-1; x<=p[0]+1; x++) {
				initGrid[x][0] = 1.0;
				initGrid[x][1] = 0.0;
				initGrid[x][2] = 0.0;
			}
		}

		output(initGrid,filename);
	}

	if (dim==2) {
		grid<2,vector<double> > initGrid(3,0,128,0,128);

		for (int i=0; i<nodes(initGrid); i++) {
			vector<int> x = position(initGrid,i);
			initGrid(i)[0] = 0.0;
			initGrid(i)[1] = 0.0;
			double d = sqrt(pow(64.0-x[0],2)+pow(64.0-x[1],2));
			if (d<32.0) initGrid(i)[2] = 1.0;
			else initGrid(i)[1] = 1.0;
		}

		for (int j=0; j<50; j++) {
			int i = rand()%nodes(initGrid);
			vector<int> p = position(initGrid,i);
			for (int x=p[0]-1; x<=p[0]+1; x++)
				for (int y=p[1]-1; y<=p[1]+1; y++) {
					initGrid[x][y][0] = 1.0;
					initGrid[x][y][1] = 0.0;
					initGrid[x][y][2] = 0.0;
				}
		}

		output(initGrid,filename);
	}

	if (dim==3) {
		grid<3,vector<double> > initGrid(3,0,64,0,64,0,64);

		for (int i=0; i<nodes(initGrid); i++) {
			vector<int> x = position(initGrid,i);
			initGrid(i)[0] = 0.0;
			initGrid(i)[1] = 0.0;
			double d = sqrt(pow(32.0-x[0],2)+pow(32.0-x[1],2)+pow(32.0-x[2],2));
			if (d<16.0) initGrid(i)[2] = 1.0;
			else initGrid(i)[1] = 1.0;
		}

		for (int j=0; j<50; j++) {
			int i = rand()%nodes(initGrid);
			vector<int> p = position(initGrid,i);
			for (int x=p[0]-1; x<=p[0]+1; x++)
				for (int y=p[1]-1; y<=p[1]+1; y++)
					for (int z=p[1]-1; z<=p[2]+1; z++) {
						initGrid[x][y][z][0] = 1.0;
						initGrid[x][y][z][1] = 0.0;
						initGrid[x][y][z][2] = 0.0;
					}
		}

		output(initGrid,filename);
	}
}

template <int dim> void update(grid<dim,vector<double> >& oldGrid, int steps)
{
	grid<dim,vector<double> > newGrid(oldGrid);

	double dt = 0.01;
	double width = 8.0;

	for (int step=0; step<steps; step++) {
		for (int i=0; i<nodes(oldGrid); i++) {
			// determine nonzero fields within
			// the neighborhood of this node
			double S = 0.0;
			vector<int> s(fields(oldGrid),0);
			vector<int> x = position(oldGrid,i);

			for (int h=0; h<fields(oldGrid); h++) {
				for (int j=0; j<dim; j++)
					for (int k=-1; k<=1; k++) {
						x[j] += k;
						if (oldGrid(x)[h]>0.0) {
							s[h] = 1;
							x[j] -= k;
							goto next;
						}
						x[j] -= k;
					}
				next: S += s[h];
			}

			// if only one field is nonzero,
			// then copy this node to newGrid
			if (S<2.0) newGrid(i) = oldGrid(i);

			else {
				// compute laplacian of each field
				vector<double> lap = laplacian(oldGrid,i);

				// compute variational derivatives
				vector<double> dFdp(fields(oldGrid),0.0);
				for (int h=0; h<fields(oldGrid); h++)
					if (s[h]>0.0)
						for (int j=h+1; j<fields(oldGrid); j++)
							if (s[j]>0.0) {
								double gamma = energy(h,j);
								double eps = 4.0/acos(-1.0)*sqrt(0.5*gamma*width);
								double w = 4.0*gamma/width;
								dFdp[h] += 0.5*eps*eps*lap[j]+w*oldGrid(i)[j];
								dFdp[j] += 0.5*eps*eps*lap[h]+w*oldGrid(i)[h];
								for (int k=j+1; k<fields(oldGrid); k++)
									if (s[k]>0.0) {
										dFdp[h] += 3.0*oldGrid(i)[j]*oldGrid(i)[k];
										dFdp[j] += 3.0*oldGrid(i)[k]*oldGrid(i)[h];
										dFdp[k] += 3.0*oldGrid(i)[h]*oldGrid(i)[j];
									}
							}

				// compute time derivatives
				vector<double> dpdt(fields(oldGrid),0.0);
				for (int h=0; h<fields(oldGrid); h++)
					if (s[h]>0.0)
						for (int j=h+1; j<fields(oldGrid); j++)
							if (s[j]>0.0) {
								double mu = mobility(h,j);
								// set mobility of particles to zero
								if (h==0 or j==0) mu = 0.0;
								dpdt[h] -= mu*(dFdp[h]-dFdp[j]);
								dpdt[j] -= mu*(dFdp[j]-dFdp[h]);
							}

				// compute newGrid values
				double sum = 0.0;
				for (int h=0; h<fields(oldGrid); h++) {
					double value = oldGrid(i)[h]+dt*(2.0/S)*dpdt[h];
					if (value>1.0) value = 1.0;
					if (value<0.0) value = 0.0;
					newGrid(i)[h] = value;
					sum += value;
				}

				// project onto Gibbs simplex
				double rsum = 0.0;
				if (fabs(sum)>0.0) rsum = 1.0/sum;
				for (int h=0; h<fields(oldGrid); h++)
					newGrid(i)[h] *= rsum;
			}
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}
}

} // namespace MMSP

#endif

#include"MMSP.main.hpp"
