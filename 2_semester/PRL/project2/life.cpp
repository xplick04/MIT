/*  
* Project:  PRL - project 2 
* Author:   Maxim Plička (xplick04, 231813)
* Date:     2024-04-05
*/

#include <iostream>
#include <mpi.h>
#include <vector>

#define ALIVE 1
#define DEAD 0

#define TOP 0
#define BOTTOM 1

struct BoardInfo
{
    int rows = 0;
    int cols = 0;
};

class Processor 
{
    private:
    std::vector<int> board; //only for rank 0
    std::vector<std::vector<int>> boardPart;
    std::vector<std::vector<int>> neighbourRows;
    BoardInfo boardInfo;
    int rank;
    int size;

    public:
    Processor(int rank, int size) : rank(rank), size(size) 
    {
        boardInfo.cols = 0;
        boardInfo.rows = 0;
    }

    void printBoard()
    {
        for (int i = 0; i < boardPart.size(); i++)
        {
            std::cout << rank << ": ";

            for (int j = 0; j < boardPart[i].size(); j++)
            {
                std::cout << boardPart[i][j];
            }
            std::cout << std::endl;
        }
        
    }

    void readFile(std::string inputFile)
    {
        FILE *f;

        f = fopen(inputFile.c_str(), "r");
        char c = fgetc(f);
        while(c != EOF)
        {
            switch (c)
            {
                case '\n':
                    boardInfo.rows++;
                    break;

                case '0':
                    if (boardInfo.rows == 0) boardInfo.cols++;
                    board.push_back(0);
                    break;

                case '1':     
                    if (boardInfo.rows == 0) boardInfo.cols++;    
                    board.push_back(1);
                    break;
            }
            c = fgetc(f);
        }
        fclose(f);

    }

    void distributeData()
    {
        int boardSize = board.size();
        MPI_Bcast(&boardSize, 1, MPI_INT, 0, MPI_COMM_WORLD); // first process board size
        MPI_Bcast(&boardInfo.cols, 1, MPI_INT, 0, MPI_COMM_WORLD); // then cols
        MPI_Bcast(&boardInfo.rows, 1, MPI_INT, 0, MPI_COMM_WORLD); // then rows
        board.resize((boardSize / size) + 1);   // + 1 for remainder

        std::vector<int> scatterCounts(size);
        int remainder = boardInfo.rows % size;
        int quotient = boardInfo.rows / size;
        for (int i = 0; i < size; ++i) 
        {
            scatterCounts[i] = (i < remainder) ? (quotient + 1) * boardInfo.cols : quotient * boardInfo.cols;
        }

        std::vector<int> offsets(size);
        int offset = 0;
        for (int i = 0; i < size; i++)
        {   
            offsets[i] = offset;
            offset += scatterCounts[i];
        }

        std::vector<int> receivedData(scatterCounts[rank]);
        MPI_Scatterv(board.data(), scatterCounts.data(), offsets.data(), MPI_INT , receivedData.data(), scatterCounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

        deserializeData(receivedData);
    }

    void deserializeData(std::vector<int> receivedData)
    {
        boardPart.clear();
        std::vector<int> row;
        for (int i = 0; i < receivedData.size(); i++)
        {
            row.push_back(receivedData[i]);
            if(row.size() == boardInfo.cols)
            {
                boardPart.push_back(row);
                row.clear();
            }
        }
        board.clear();
    }

    void computeNextState()
    {
        std::vector<std::vector<int>> newBoardPart = boardPart;
        getNeighbourRows();
        int numOfNeighbours = 0;

        for(int i = 0; i < boardPart.size(); i++)
        {
            for(int j = 0; j < boardPart[i].size(); j++)
            {
                numOfNeighbours = computeNumOfNeighbours(i, j);
                newBoardPart[i][j] = liveOrDie(boardPart[i][j], numOfNeighbours);
            }
        }
        
        boardPart = newBoardPart;
    }

    int computeNumOfNeighbours(int row, int col)
    {
        int numOfNeighbours = 0;
        std::vector<std::vector<int>> tmp;   // selector between boardPart and neighbourRows

        for (int i = 0; i < 9; i++)
        {
            if (i == 4) continue;   // skip the cell itself
            tmp = boardPart;
            int neighbourRow = row + i / 3 - 1; // indexation for neighbour
            int neighbourCol = col + i % 3 - 1; // indexation for neighbour

            if(neighbourRow < 0) // upper neighbour
            {
                neighbourRow = TOP;
                tmp = neighbourRows;
            }
            else if(neighbourRow >= boardPart.size()) // lower neighbour
            {
                neighbourRow = BOTTOM;
                tmp = neighbourRows;
            }

            if(neighbourCol < 0) // left neighbour
            {
                neighbourCol = boardPart[0].size() - 1;
            }
            else if(neighbourCol >= boardPart[0].size()) // right neighbour    
            {
                neighbourCol = 0;
            }
            numOfNeighbours += tmp[neighbourRow][neighbourCol];
        }
        return numOfNeighbours;
    }

    void getNeighbourRows()
    {
        neighbourRows.clear();
        std::vector<int> receivedDataUpper(boardInfo.cols);
        std::vector<int> receivedDataLower(boardInfo.cols);
        MPI_Request send_request, recv_request;
        MPI_Status send_status, recv_status;
        int upperNeighbour = (rank == 0) ? size - 1 : rank - 1;
        int lowerNeighbour = (rank == size - 1) ? 0 : rank + 1;
        
        // Non-blocking send to the upper neighbor
        MPI_Isend(boardPart[0].data(), boardInfo.cols, MPI_INT, upperNeighbour, 0, MPI_COMM_WORLD, &send_request);

        // Non-blocking receive from the lower neighbor
        MPI_Irecv(receivedDataLower.data(), boardInfo.cols, MPI_INT, lowerNeighbour, 0, MPI_COMM_WORLD, &recv_request);

        // Wait for both send and receive operations to complete
        MPI_Wait(&send_request, &send_status);
        MPI_Wait(&recv_request, &recv_status);

        // Non-blocking send to the upper neighbor
        MPI_Isend(boardPart[boardPart.size() - 1].data(), boardInfo.cols, MPI_INT, lowerNeighbour, 0, MPI_COMM_WORLD, &send_request);

        // Non-blocking receive from the lower neighbor
        MPI_Irecv(receivedDataUpper.data(), boardInfo.cols, MPI_INT, upperNeighbour, 0, MPI_COMM_WORLD, &recv_request);

        // Wait for both send and receive operations to complete
        MPI_Wait(&send_request, &send_status);
        MPI_Wait(&recv_request, &recv_status);

        neighbourRows.push_back(receivedDataUpper);
        neighbourRows.push_back(receivedDataLower);
        
    }

    int liveOrDie(int cellStatus, int numOfNeighbours)
    {
        if(cellStatus == ALIVE)
        {
            if(numOfNeighbours < 2 || numOfNeighbours > 3)
            {
                return DEAD;
            }
            else
            {
                return ALIVE;
            }
        }
        else
        {
            if(numOfNeighbours == 3)
            {
                return ALIVE;
            }
            else
            {
                return DEAD;
            }
        }
    }

};


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    if (argc < 3) 
    {
        std::cerr << "Usage: sh test.sh <input_file> <steps>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int steps = std::stoi(argv[2]);
    Processor *p = new Processor(rank, size);

    if (rank == 0) 
    {
        std::string inputFile = argv[1];
        p->readFile(inputFile);
    }
    p->distributeData();

    for(int i = 0; i < steps; i++)
    {
        p->computeNextState();
        MPI_Barrier(MPI_COMM_WORLD);    // wait for all processes to finish computing
    }
    p->printBoard();
    
    MPI_Finalize();
    return 0;
}
