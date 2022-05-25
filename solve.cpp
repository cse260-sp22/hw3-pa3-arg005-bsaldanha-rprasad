/*
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 *
 * Modified and  restructured by Scott B. Baden, UCSD
 *
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <cstring>
#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
void printArray(double *E, int m);
void printArrayInt(int *E, int m);

extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq) {
    double l2norm = sumSq / (double)((cb.m) * (cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

inline int allotDimension(const int n, const int nprocs, const int rank) {
    // returns 1-d allocation of total n resources among nprocs to the process with rank `rank`
    int result = n / nprocs;
    if (rank < n % nprocs) {
        result++;
    }
    return result;
}

inline int dimensionOffset(const int n, const int nprocs, const int rank) {
    if (rank < n % nprocs) {
        return rank * (n / nprocs + 1);
    }
    return (n / nprocs) * rank + (n % nprocs);
}

inline void setCurrentProcessorDim(int &m, int &n, const int rank, const int nprocs) {
    // sets m, n to the tile's dimensions of the current processor
    m = allotDimension(cb.m, cb.py, rank / cb.px);
    n = allotDimension(cb.n, cb.px, rank % cb.px);
}


void fillSendCounts(int *sendcounts, const int nprocs) {
    int rank, mproc, nproc, size;
    for (rank = 0; rank < nprocs; ++rank) {
        setCurrentProcessorDim(mproc, nproc, rank, nprocs);
        size = mproc * nproc;
        sendcounts[rank] = size;
    }
}

void fillSendDispls(int *senddispls, const int nprocs) {
    int rank, mproc, nproc;
    senddispls[0] = 0;
    for (rank = 1; rank < nprocs; ++rank) {
        setCurrentProcessorDim(mproc, nproc, rank - 1, nprocs);
        senddispls[rank] = senddispls[rank - 1] + mproc * nproc;
    }
}

void repackForScattering(double *data, double *packed, const int nprocs) {
    int m, n, rowOffset, colOffset, idx = 0, row, col;
    for (int rank = 0; rank < nprocs; ++rank) {
        // for each rank, identify where to start packing data!
        // gets m and n for `rank`
        setCurrentProcessorDim(m, n, rank, nprocs);
        // gets row and column offsets for `rank`
        rowOffset = dimensionOffset(cb.m, cb.py, rank / cb.px);
        colOffset = dimensionOffset(cb.n, cb.px, rank % cb.px);
        // copy row by row (for m columns)
        for (int i = 0; i < m; ++i) {
            row = rowOffset + i;
            col = colOffset + 1;
            cout << "row = " << row << " col = " << col << endl;
            cout << "copying from " << row * (cb.n + 2) + col << " to " << row * (cb.n + 2) + col + n << endl;
            memcpy(packed + idx * n, data + row * (cb.n + 2) + col, n * sizeof(double));
        }
    }
}

inline void scatterInitialCondition(double *E, double *R, const int nprocs, const int myrank, const int m, const int n) {
    const int receiveCount = m * n;

    int *sendcounts = new int[nprocs];
    int *senddispls = new int[nprocs];

    double* sendE = new double[cb.m * cb.n];
    double* sendR = new double[cb.m * cb.n];

    printMat2("E",E,cb.m, cb.n);
    printMat2("R",R,cb.m, cb.n);

    repackForScattering(E, sendE, nprocs);
    repackForScattering(R, sendR, nprocs);

    fillSendCounts(sendcounts, nprocs);
    fillSendDispls(senddispls, nprocs);

    if (myrank == 0) {
        cout << "sendcounts: ";
        printArrayInt(sendcounts, nprocs);
        cout << "\n";

        cout << "senddispls: ";
        printArrayInt(senddispls, nprocs);
        cout << "\n";

        cout << "sendE: ";
        printArray(sendE, cb.m * cb.n);
        cout << "\n";

        cout << "sendR: ";
        printArray(sendR, cb.m * cb.n);
        cout << "\n";
    }

    MPI_Scatterv(sendE, sendcounts, senddispls, MPI_DOUBLE, R, receiveCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(sendR, sendcounts, senddispls, MPI_DOUBLE, E, receiveCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void solveMPIArpit(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {

    #ifndef _MPI_
        cout << "Error: MPI not enabled" << endl;
        return;
    #endif

    // MPI_Init(&argc,&argv);

    int nprocs = 1, myrank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;

    // initializing current processor dimensions
    int m, n; // dimensions of E current processor would compute
    setCurrentProcessorDim(m, n, myrank, nprocs);
    // there would be padding of 2 as well. TODO!

    // offsets in terms of processors
    int rowOffset = dimensionOffset(cb.m, cb.py, myrank / cb.px);
    int colOffset = dimensionOffset(cb.n, cb.px, myrank % cb.px);

    int innerBlockRowStartIndex = rowOffset * (cb.n + 2) + (colOffset + 1);
    int innerBlockRowEndIndex = innerBlockRowStartIndex + m * (cb.n + 2);

    if (cb.debug) {
        cout << "Processor " << myrank << ": " << "m = " << m << ", n = " << n << ", rowOffset = " << rowOffset << ", colOffset = " << colOffset << ", innerBlockRowStartIndex = " << innerBlockRowStartIndex << ", innerBlockRowEndIndex = " << innerBlockRowEndIndex << endl;
    }

    scatterInitialCondition(E_prev, R, nprocs, myrank, m, n);
    return;

    // scatter the initial conditions to all the other processes
    // MPI_Sendrecv(
    // );

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    for (niter = 0; niter < cb.niters; niter++) {

        if (cb.debug && (niter == 0)) {
            stats(E_prev, m, n, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, m, n, -1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E, -1, m + 1, n + 1);
        }

        /*
         * Copy data from boundary of the computational box to the
         * padding region, set up for differencing computational box's boundary
         *
         * These are physical boundary conditions, and are not to be confused
         * with ghost cells that we would use in an MPI implementation
         *
         * The reason why we copy boundary conditions is to avoid
         * computing single sided differences at the boundaries
         * which increase the running time of solve()
         *
         */

        // 4 FOR LOOPS set up the padding needed for the boundary conditions
        int i, j;

        // Fills in the TOP Ghost Cells
        for (i = 0; i < (n + 2); i++) {
            E_prev[i] = E_prev[i + (n + 2) * 2];
        }

        // Fills in the RIGHT Ghost Cells
        for (i = (n + 1); i < (m + 2) * (n + 2); i += (n + 2)) {
            E_prev[i] = E_prev[i - 2];
        }

        // Fills in the LEFT Ghost Cells
        for (i = 0; i < (m + 2) * (n + 2); i += (n + 2)) {
            E_prev[i] = E_prev[i + 2];
        }

        // Fills in the BOTTOM Ghost Cells
        for (i = ((m + 2) * (n + 2) - (n + 2)); i < (m + 2) * (n + 2); i++) {
            E_prev[i] = E_prev[i - (n + 2) * 2];
        }

        //////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2)) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            for (i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
            }
        }
#else
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2)) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
            }
        }

        /*
         * Solve the ODE, advancing excitation and recovery variables
         *     to the next timtestep
         */

        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2)) {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < n; i++) {
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
            }
        }
#endif
        /////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq) {
            if (!(niter % cb.stats_freq)) {
                stats(E, m, n, &mx, &sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm, mx, dt, m, n, niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq) {
            if (!(niter % cb.plot_freq)) {
                plotter->updatePlot(E, niter, m, n);
            }
        }

        // Swap current and previous meshes
        double *tmp = E;
        E = E_prev;
        E_prev = tmp;

    } // end of 'niter' loop at the beginning

    //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

    stats(E_prev, m, n, &Linf, &sumSq);
    L2 = L2Norm(sumSq);

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}
 

void solveOriginal(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {

    // Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;
    int m = cb.m, n = cb.n;
    int innerBlockRowStartIndex = (n + 2) + 1;
    int innerBlockRowEndIndex = (((m + 2) * (n + 2) - 1) - (n)) - (n + 2);

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    for (niter = 0; niter < cb.niters; niter++) {

        if (cb.debug && (niter == 0)) {
            stats(E_prev, m, n, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, m, n, -1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E, -1, m + 1, n + 1);
        }

        /*
         * Copy data from boundary of the computational box to the
         * padding region, set up for differencing computational box's boundary
         *
         * These are physical boundary conditions, and are not to be confused
         * with ghost cells that we would use in an MPI implementation
         *
         * The reason why we copy boundary conditions is to avoid
         * computing single sided differences at the boundaries
         * which increase the running time of solve()
         *
         */

        // 4 FOR LOOPS set up the padding needed for the boundary conditions
        int i, j;

        // Fills in the TOP Ghost Cells
        for (i = 0; i < (n + 2); i++) {
            E_prev[i] = E_prev[i + (n + 2) * 2];
        }

        // Fills in the RIGHT Ghost Cells
        for (i = (n + 1); i < (m + 2) * (n + 2); i += (n + 2)) {
            E_prev[i] = E_prev[i - 2];
        }

        // Fills in the LEFT Ghost Cells
        for (i = 0; i < (m + 2) * (n + 2); i += (n + 2)) {
            E_prev[i] = E_prev[i + 2];
        }

        // Fills in the BOTTOM Ghost Cells
        for (i = ((m + 2) * (n + 2) - (n + 2)); i < (m + 2) * (n + 2); i++) {
            E_prev[i] = E_prev[i - (n + 2) * 2];
        }

        //////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2)) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            for (i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
            }
        }
#else
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2)) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
            }
        }

        /*
         * Solve the ODE, advancing excitation and recovery variables
         *     to the next timtestep
         */

        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2)) {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < n; i++) {
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
            }
        }
#endif
        /////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq) {
            if (!(niter % cb.stats_freq)) {
                stats(E, m, n, &mx, &sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm, mx, dt, m, n, niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq) {
            if (!(niter % cb.plot_freq)) {
                plotter->updatePlot(E, niter, m, n);
            }
        }

        // Swap current and previous meshes
        double *tmp = E;
        E = E_prev;
        E_prev = tmp;

    } // end of 'niter' loop at the beginning

    //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

    stats(E_prev, m, n, &Linf, &sumSq);
    L2 = L2Norm(sumSq);

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}


void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {
    solveMPIArpit(_E, _E_prev, R, alpha, dt, plotter, L2, Linf);
} 

void printMat2(const char mesg[], double *E, int m, int n)
{
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m > 34)
        return;
#endif
    printf("%s\n", mesg);
    for (i = 0; i < (m + 2) * (n + 2); i++) {
        int rowIndex = i / (n + 2);
        int colIndex = i % (n + 2);
        if ((colIndex > 0) && (colIndex < n + 1))
            if ((rowIndex > 0) && (rowIndex < m + 1))
                printf("%6.3f ", E[i]);
        if (colIndex == n + 1)
            printf("\n");
    }
}
