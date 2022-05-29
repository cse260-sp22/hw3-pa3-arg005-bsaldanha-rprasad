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
#include <vector>
#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
void printArray(double *E, int m);
void printArrayInt(int *E, int m);
double *alloc1D(int m,int n);

enum Direction { LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3 };

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
            row = rowOffset + 1 + i;
            col = colOffset + 1;
            memcpy(packed + idx, data + row * (cb.n + 2) + col, n * sizeof(double));
            idx += n;
        }
    }
}

void repackForGathering(double *data, double *packed, const int nprocs, const int myrank) {
    int m, n, rowOffset, colOffset, idx = 0, row, col;

	setCurrentProcessorDim(m, n, myrank, nprocs);
	// copy row by row (for m columns)
	for (int i = 0; i < m; ++i) {
		row =  1 + i;
		col = 1;
		memcpy(packed + idx, data + row * (n + 2) + col, n * sizeof(double));
		idx += n;
	}
}

void unpackForPlotting(double *data, double *unpacked, const int nprocs) {
	int m, n, rowOffset, colOffset, idx = 0, row, col;

    for (int rank = 0; rank < nprocs; ++rank) {
        // for each rank, identify where to start unpacking data!
        // gets m and n for `rank`
        setCurrentProcessorDim(m, n, rank, nprocs);
        // gets row and column offsets for `rank`
        rowOffset = dimensionOffset(cb.m, cb.py, rank / cb.px);
        colOffset = dimensionOffset(cb.n, cb.px, rank % cb.px);
        // copy row by row (for m columns)
        for (int i = 0; i < m; ++i) {
            row = rowOffset + i;
            col = colOffset;
            // cout << "row = " << row << " col = " << col << endl;
            // cout << "copying from " << row * (cb.n + 2) + col << " to " << row * (cb.n + 2) + col + n << endl;
            memcpy(unpacked + row * cb.n + col, data + idx, n * sizeof(double));
            idx += n;
            // cout << "packed array: from " << idx * n << ": ";
            // printArray(packed + idx, n);
        }
    }
}

inline void sendReceive(const int m, const int n, Direction direction, double *data, const int myrank, vector<MPI_Request>& requests) {
    int send_index, receive_index, otherProcessRank;

    switch(direction) {
        case LEFT:
            send_index = 1 + (n + 2);
            receive_index = (n + 2);
            otherProcessRank = myrank - 1;
            break;
        case RIGHT:
            send_index = n + (n + 2);
            receive_index = n + 1 + (n + 2);
            otherProcessRank = myrank + 1;
            break;
        case UP:
            send_index = 1 + (n + 2);
            receive_index = 1;
            otherProcessRank = myrank - cb.px;
            break;
        case DOWN:
            send_index = (n + 2) * (m + 2) - 2 * (n + 2) + 1;
            receive_index = (n + 2) * (m + 2) - (n + 2) + 1;
            otherProcessRank = myrank + cb.px;
            break;
    }


    MPI_Datatype left_right_boundary_col;

    // vector to send left and right boundaries
    MPI_Type_vector(m, 1, (n + 2), MPI_DOUBLE, &left_right_boundary_col);

    MPI_Request send_request, recv_request;

    if (direction == LEFT || direction == RIGHT) {
        // send left and right boundaries
        MPI_Isend(data + send_index, 1, left_right_boundary_col, otherProcessRank, direction, MPI_COMM_WORLD, &send_request);
        MPI_Irecv(data + receive_index, 1, left_right_boundary_col, otherProcessRank, direction, MPI_COMM_WORLD, &recv_request);
    } else {
        // send top and bottom boundaries
        MPI_Isend(data + send_index, n, MPI_DOUBLE, otherProcessRank, direction, MPI_COMM_WORLD, &send_request);
        MPI_Irecv(data + receive_index, n, MPI_DOUBLE, otherProcessRank, direction, MPI_COMM_WORLD, &recv_request);
    }

    requests.push_back(send_request);
    requests.push_back(recv_request);
}

void communicateGhostCells(const int m, const int n, double *data, const int my_rank, vector<MPI_Request>& requests) {
    // l = 0, r = 1, u = 2, d = 3
    // compute inner cells and populate ghost cells in parallel
    // once the inner cells are computed, `wait` for all ghost cells to be populated
    // compute outer cells
    // data could be E_prev or R
    int row = my_rank / cb.px;
    int col = my_rank % cb.px;

    bool communicateLeft = col == 0;
    bool communicateRight = col == cb.px - 1;
    bool communicateUp = row == 0;
    bool communicateDown = row == cb.py - 1;

    if (communicateLeft) {
        sendReceive(m, n, LEFT, data, my_rank, requests);
    }

    if (communicateRight) {
        sendReceive(m, n, RIGHT, data, my_rank, requests);
    }

    if (communicateUp) {
        sendReceive(m, n, UP, data, my_rank, requests);
    }

    if (communicateDown) {
        sendReceive(m, n, DOWN, data, my_rank, requests);
    }
}

// helper ode/pde functions
inline void applyODE(double *E_tmp, double *E_prev_tmp, double *R_tmp, const int i, const int m, const int n, const double alpha) {
    // i: cell index
    E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
}

inline void applyPDE(double *E_tmp, double *E_prev_tmp, double *R_tmp, const int i, const int m, const int n, const double dt) {
    E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
    R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
}

inline void applyODEPDE(double *E_tmp, double *E_prev_tmp, double *R_tmp, const int i, const int m, const int n, const double dt, const double alpha) {
    applyODE(E_tmp, E_prev_tmp, R_tmp, i, m, n, alpha);
    applyPDE(E_tmp, E_prev_tmp, R_tmp, i, m, n, dt);
}

inline void compute(const int m, const int n, const double dt, const double alpha, double *E, double *E_tmp, double *E_prev, double *E_prev_tmp, double *R, double *R_tmp, const int my_rank) {
    vector<MPI_Request> requests;
    requests = communicateGhostCells(m, n, E_prev, my_rank, requests);

    // this computes interior
    int interior_start_row = 1 + 2 * (n + 2); // 1 + 2*rows b/c in fused cell's for loop, i goes from 2 to n - 1
    int interior_end_row = (n + 2) * (m + 2) - 3 * (n + 2) + 1;

    for(int niter = 0; niter < cb.niters; niter++) {
        #define FUSED 1

        #ifdef FUSED
                // Solve for the excitation, a PDE
                for(int j = interior_start_row; j <= interior_end_row; j += (n + 2)) {
                    E_tmp = E + j;
                    E_prev_tmp = E_prev + j;
                    R_tmp = R + j;
                    for (int i = 2; i < n; i++) {
                        applyODEPDE(E_tmp, E_prev_tmp, R_tmp, i, m, n, dt, alpha);
                        // E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
                        // E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                        // R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
                    }
                }
        #else
                // Solve for the excitation, a PDE
                for(int j = interior_start_row; j < interior_end_row; j += (n + 2)) {
                    E_tmp = E + j;
                    E_prev_tmp = E_prev + j;
                    for (int i = 2; i < n; i++) {
                        applyODE(E_tmp, E_prev_tmp, R_tmp, i, m, n, alpha);
                        // E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
                    }
                }

                /*
                * Solve the ODE, advancing excitation and recovery variables
                *     to the next timtestep
                */

                for(int j = interior_start_row; j < interior_end_row; j += (n + 2)) {
                    E_tmp = E + j;
                    E_prev_tmp = E_prev + j;
                    R_tmp = R + j;
                    for (int i = 2; i < n; i++) {
                        applyPDE(E_tmp, E_prev_tmp, R_tmp, i, m, n, dt);
                        // E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                        // R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
                    }
                }
        #endif
    }

    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);

    // compute boundary cells
    for(int niter = 0; niter < cb.niters; niter++) {
        #define FUSED 1

        #ifdef FUSED
                // Solve for the excitation, a PDE

                // first row
                int row_offset = 1 + (n + 2);
                E_tmp = E + row_offset;
                E_prev_tmp = E_prev + row_offset;
                R_tmp = R + row_offset;

                for (int i = 1; i <= n; i++) {
                    applyODEPDE(E_tmp, E_prev_tmp, R_tmp, i, m, n, dt, alpha);
                }

                // last row
                row_offset = (n + 2) * (m + 2) - 2 * (n + 2) + 1;
                E_tmp = E + row_offset;
                E_prev_tmp = E_prev + row_offset;
                R_tmp = R + row_offset;

                for (int i = 1; i <= n; i++) {
                    applyODEPDE(E_tmp, E_prev_tmp, R_tmp, i, m, n, dt, alpha);
                }

                // first col
                int start_index = 1 + (n + 2);
                int end_index = ((n + 2) * (m + 2) - 2 * (n + 2) + 1);

                for (int i = start_index; i <= end_index; i += (n + 2)) {
                    applyODEPDE(E, E_prev, R, i, m, n, dt, alpha);
                }

                // last col
                start_index = n + (n + 2);
                end_index = ((n + 2) * (m + 2) - 2 * (n + 2) + n) - row_offset;

                for (int i = start_index; i <= end_index; i += (n + 2)) {
                    applyODEPDE(E, E_prev, R, i, m, n, dt, alpha);
                }

        #else
                // Solve for the excitation, a PDE
                // first row
                row_offset = 1 + (n + 2);
                E_tmp = E + row_offset;
                E_prev_tmp = E_prev + row_offset;
                R_tmp = R + row_offset;

                for (int i = 1; i <= n; i++) {
                    applyODE(E_tmp, E_prev_tmp, R_tmp, i, m, n, alpha);
                }

                /*
                * Solve the ODE, advancing excitation and recovery variables
                *     to the next timtestep
                */

                for (int i = 1; i <= n; i++) {
                    applyPDE(E, E_prev, R, i, m, n, dt);
                }

                // last row
                row_offset = (n + 2) * (m + 2) - 2 * (n + 2) + 1;
                E_tmp = E + row_offset;
                E_prev_tmp = E_prev + row_offset;
                R_tmp = R + row_offset;

                for (int i = 1; i <= n; i++) {
                    applyODE(E_tmp, E_prev_tmp, R_tmp, i, m, n, alpha);
                }

                /*
                * Solve the ODE, advancing excitation and recovery variables
                *     to the next timtestep
                */

                for (int i = 1; i <= n; i++) {
                    applyPDE(E, E_prev, R, i, m, n, dt);
                }

                // first col
                int start_index = 1 + (n + 2);
                int end_index = ((n + 2) * (m + 2) - 2 * (n + 2) + 1);

                for (int i = start_index; i <= end_index; i += (n + 2)) {
                    applyODE(E, E_prev, R, i, m, n, alpha);
                }

                /*
                * Solve the ODE, advancing excitation and recovery variables
                *     to the next timtestep
                */

                for (int i = start_index; i <= end_index; i += (n + 2)) {
                    applyPDE(E, E_prev, R, i, m, n, dt);
                }

                // last col
                start_index = n + (n + 2);
                end_index = ((n + 2) * (m + 2) - 2 * (n + 2) + n) - row_offset;

                for (int i = start_index; i <= end_index; i += (n + 2)) {
                    applyODE(E, E_prev, R, i, m, n, alpha);
                }

                for (int i = start_index; i <= end_index; i += (n + 2)) {
                    applyPDE(E, E_prev, R, i, m, n, dt);
                }
        #endif
    }
}

inline void scatterInitialCondition(
    double *E, double *R, const int nprocs, const int myrank, const int m, const int n,
    double *recvE, double *recvR
) {
    const int receiveCount = m * n;

    int *sendcounts = new int[nprocs];
    int *senddispls = new int[nprocs];

    double* sendE = alloc1D(cb.m, cb.n);
    double* sendR = alloc1D(cb.m, cb.n);

    repackForScattering(E, sendE, nprocs);
    repackForScattering(R, sendR, nprocs);

    fillSendCounts(sendcounts, nprocs);
    fillSendDispls(senddispls, nprocs);

    if (myrank == 0 && cb.debug) {
        cout << "sendcounts: ";
        printArrayInt(sendcounts, nprocs);
        cout << "\n";

        cout << "senddispls: ";
        printArrayInt(senddispls, nprocs);
        cout << "\n";

        cout << "sendE: ";
        printArray(sendE, (cb.m + 2) * (cb.n + 2));
        cout << "\n";

        cout << "sendR: ";
        printArray(sendR, (cb.m + 2) * (cb.n + 2));
        cout << "\n";
    }

    double* s_tempE = alloc1D(m, n);
    double* s_tempR = alloc1D(m, n);

    MPI_Scatterv(sendE, sendcounts, senddispls, MPI_DOUBLE, s_tempE, receiveCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(sendR, sendcounts, senddispls, MPI_DOUBLE, s_tempR, receiveCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // free(sendE);
    // free(sendR);
    // free(sendcounts);
    // free(senddispls);

    memset(recvE, 0.0, (m + 2) * (n + 2) * sizeof(double));
    memset(recvR, 0.0, (m + 2) * (n + 2) * sizeof(double));

    for (int j = 0; j < m; ++j) {
        memcpy(recvE + (n + 2) * (j + 1) + 1, s_tempE + j * n, n * sizeof(double));
        memcpy(recvR + (n + 2) * (j + 1) + 1, s_tempR + j * n, n * sizeof(double));
    }
}

//TODO: Code gather algo, send all data in buffer and then unpack it to get final matrix
inline void gatherFinalValues(
    double *E, double *R, const int nprocs, const int myrank, const int m, const int n,
    double *recvE, double *recvR
) {
    const int receiveCount = cb.m * cb.n;

    int *sendcounts = new int[nprocs];
    int *senddispls = new int[nprocs];

	double* finE = alloc1D(cb.m, cb.n);
	double* finR = alloc1D(cb.m, cb.n);

	double* finE_unpacked = alloc1D(cb.m, cb.n);
	double* finR_unpacked = alloc1D(cb.m, cb.n);

    // printMat2("E",E,cb.m, cb.n);
    // printMat2("R",R,cb.m, cb.n);
	
    double* r_tempE = alloc1D(m, n);
    double* r_tempR = alloc1D(m, n);
    
	repackForGathering(recvE, r_tempE, nprocs, myrank);
    repackForGathering(recvR, r_tempR, nprocs, myrank);

    fillSendCounts(sendcounts, nprocs);
    fillSendDispls(senddispls, nprocs);

    //if (myrank == 0 && cb.debug) {
    //    cout << "sendcounts: ";
    //    printArrayInt(sendcounts, nprocs);
    //    cout << "\n";

    //    cout << "senddispls: ";
    //    printArrayInt(senddispls, nprocs);
    //    cout << "\n";

    //    cout << "sendE: ";
    //    printArray(sendE, (cb.m + 2) * (cb.n + 2));
    //    cout << "\n";

    //    cout << "sendR: ";
    //    printArray(sendR, (cb.m + 2) * (cb.n + 2));
    //    cout << "\n";
    //}

    // double* recvE = new double[receiveCount];
    // double* recvR = new double[receiveCount];

	MPI_Gatherv(r_tempE, receiveCount, MPI_DOUBLE, finE, sendcounts, senddispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv(r_tempR, receiveCount, MPI_DOUBLE, finR, sendcounts, senddispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	unpackForPlotting(finE, finE_unpacked, nprocs);
	unpackForPlotting(finR, finR_unpacked, nprocs);

    // free(sendE);
    // free(sendR);
    // free(sendcounts);
    // free(senddispls);
}

void padBoundaries(int m, int n, double *E_prev, const int myrank) {
    /*
    pad boundaries only if the processor's rank is on the corners
    UPDATE the below method!
    */

    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int i, j;

	int row = myrank / cb.px;
	int col = myrank % cb.px;

	if (row == 0){
		// Fills in  the TOP Ghost Cells
		for (i = 1; i < (n+1); i++){
			E_prev[i] = E_prev[i + (n + 2)*2];
		}
	}
	else if (row == (cb.m-1)){
		// Fills in the BOTTOM Ghost Cells
		for (i = ((m + 2)*(n + 2) - (n + 2) + 1); i < (m + 2)*(n + 2) - 1; i++){
			E_prev[i] = E_prev[i - (n + 2)*2];
		}
	}

	if (col == 0){
		// Fills in the LEFT Ghost Cells
		for (i = (n + 2); i < (m + 1)*(n + 2); i += (n + 2)){
			E_prev[i] = E_prev[i + 2];	
		}
	}
	else if (col == (cb.n-1)){
		// Fills in the RIGHT Ghost Cells
		for (i = (n + 1 + n + 2); i < (m + 1)*(n + 2); i += (n + 2)){
			E_prev[i] = E_prev[i - 2];
		}
	}
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

    // initializing current processor dimensions
    int m, n; // dimensions of E current processor would compute
    setCurrentProcessorDim(m, n, myrank, nprocs);
    // there would be padding of 2 as well. TODO!

    // offsets in terms of processors
    // int rowOffset = dimensionOffset(cb.m, cb.py, myrank / cb.px);
    // int colOffset = dimensionOffset(cb.n, cb.px, myrank % cb.px);

    // int innerBlockRowStartIndex = rowOffset * (cb.n + 2) + (colOffset + 1);
    // int innerBlockRowEndIndex = innerBlockRowStartIndex + m * (cb.n + 2);

    int innerBlockRowStartIndex = (n + 2) + 1;
    int innerBlockRowEndIndex = (((m + 2) * (n + 2) - 1) - (n)) - (n + 2);

    // if (cb.debug) {
    //     cout << "Processor " << myrank << ": " << "m = " << m << ", n = " << n << ", rowOffset = " << rowOffset << ", colOffset = " << colOffset << ", innerBlockRowStartIndex = " << innerBlockRowStartIndex << ", innerBlockRowEndIndex = " << innerBlockRowEndIndex << endl;
    // }

    double* recvEprev = alloc1D(m + 2, n + 2);
    double* recvR = alloc1D(m + 2, n + 2);
    double* recvE = alloc1D(m + 2, n + 2);
    memset(recvE, 0.0, (m + 2) * (n + 2) * sizeof(double));

    // scatter the initial conditions
    double *E_prev = *_E_prev;
    scatterInitialCondition(E_prev, R, nprocs, myrank, m, n, recvEprev, recvR);

    // Simulated time is different from the integer timestep number
    double t = 0.0;
    double *E = recvE;
    double *R = recvR;
    double *R_tmp = R;
    double *E_tmp = E;
    double *E_prev_tmp = recvEprev;
    E_prev = recvEprev;

    double mx, sumSq;
    int niter, i, j;

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

        padBoundaries(m, n, E_prev, myrank); // (TODO: Brandon)

        // communicate the boundaries with other processors (TODO: Raghav & Brandon)
        // and update compute part of the function too!
        //////////////////////////////////////////////////////////////////////////////
        compute(m, n, dt, alpha, E, E_tmp, E_prev, E_prev_tmp, R, R_tmp, myrank);

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

    // free(recvE);
    // free(recvR);
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
