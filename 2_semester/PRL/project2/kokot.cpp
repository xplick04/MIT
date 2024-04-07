/*  
* Project:  PRL - project 2 
* Author:   Maxim Plička (xplick04, 231813)
* Date:     2024-04-05
*/

#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <string>

struct BoardSize
{
    int rows = 0;
    int cols = 0;
};

void printBoard(std::vector<int> receivedData, int rank)
{
    std::cout << "Rank: " << rank << " Data: ";
    for (int i = 0; i < receivedData.size(); i++)
    {
        std::cout << receivedData[i] << " ";
    }
    std::cout << std::endl; 
}

std::vector<int> readFile(std::string inputFile)
{
    std::vector<int> boardPart;
    BoardSize boardSize;
    FILE *f;

    f = fopen(inputFile.c_str(), "r");
    char c = fgetc(f);
    while(c != EOF)
    {
        switch (c)
        {
            case '\n':
                boardSize.rows++;
                break;

            case '0':
                if (boardSize.rows == 0) boardSize.cols++;
                boardPart.push_back(0);
                break;

            case '1':     
                if (boardSize.rows == 0) boardSize.cols++;    
                boardPart.push_back(1);
                break;
        }
        c = fgetc(f);
    }
    fclose(f);

    return boardPart;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Rank: " << rank << " Size: " << size << std::endl;
    std::string inputFile = argv[1];
    
    std::vector<int> dataSend;
    if (rank == 0) {
        dataSend = readFile(inputFile);
    }

    int dataSize = dataSend.size();
    MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << dataSize << std::endl;

    std::vector<int> scatterCounts(size);
    int remainder = dataSize % size;
    int quotient = dataSize / size;

    for (int i = 0; i < size; ++i) {
        scatterCounts[i] = (i < remainder) ? quotient + 1 : quotient;
    }
    std::cout << "KOKOT:" << rank << " " << scatterCounts[rank] << std::endl;

    std::vector<int> displacements(size);
    int displacement = 0;
    for (int i = 0; i < size; ++i) {
        displacements[i] = displacement;
        displacement += scatterCounts[i];
    }

    std::vector<int> receivedData(scatterCounts[rank]);
    //scattercount - number of elements to send to each process
    // displacements - Entry i specifies the displacement (relative to sendbuf) from which to take the outgoing data to process i (integer) 
    MPI_Scatterv(dataSend.data(), scatterCounts.data(), displacements.data(), MPI_INT , receivedData.data(), scatterCounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    printBoard(receivedData, rank);

    MPI_Finalize();
    return 0;
}
