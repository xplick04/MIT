/*  
* Project:  PRL - project 1 
* Author:   Maxim Plička (xplick04, 231813)
* Date:     2024-02-29
*/

#include <iostream>
#include <mpi.h>
#include <cmath>
#include <queue>
#include <unistd.h>



// Tags for communication and indexing queues
enum Tag {
    TAG_TOP = 0,
    TAG_BOTTOM = 1,
    TAG_END = 2
};


// Class for each proccess
class Node 
{
    protected:
        std::queue<unsigned char> queueTop;
        std::queue<unsigned char> queueBottom;
        int cntTop;
        int cntBottom;
        MPI_Status status;
        int rank;
        unsigned char byte;
        bool lastNum;
        bool enoughtNumbers;
        int selector;
        int size;
        bool lastNode;

    public:
        Node(int rank, int size) : rank(rank), cntTop(pow(2, rank - 1)), cntBottom(pow(2, rank - 1)),
         lastNum(false), enoughtNumbers(false), selector(TAG_TOP), size(size) , lastNode(rank == size - 1){}

        // Main function that decides which proccess will do what
        int execute()
        {
            if(rank == 0)
            {
                if(recieveFirst()) return 1;
            }
            else 
            {
                recieve();
            }
            return 0;
        }

        // First proccess recieve
        int recieveFirst()
        {
            FILE *f;
            f = fopen("./numbers", "r");
            if(f == NULL)
            {
                std::cerr << "Error: File not found!" << std::endl;
                return 1;
            }

            bool even = true;
            while(fread(&byte, 1, 1, f) == 1)
            {
                if(even && size != 1)   // If there is only one number
                {
                    MPI_Send(&byte, 1, MPI_BYTE, 1, TAG_TOP, MPI_COMM_WORLD);
                }
                else if(size != 1)  // If there is only one number
                {
                    MPI_Send(&byte, 1, MPI_BYTE, 1, TAG_BOTTOM, MPI_COMM_WORLD);
                }
                even = !even;
                std::cout << +byte << " ";
            }

            std::cout << std::endl; //formating

            if(fclose(f) != 0)
            {
                std::cerr << "Error: File not closed!" << std::endl;
                return 1;
            }

            if(size != 1)   // if there are atleast two proccesses
            {
                MPI_Send(&byte, 1, MPI_BYTE, 1, TAG_END, MPI_COMM_WORLD); // End of communication
            }
            else    // If there is only one number (one proccess generated)
            {
                std::cout << +byte <<std::endl;
            }

            // Recieve numbers for printing from last node
            while(true && size != 1)
            {
                MPI_Recv(&byte, 1, MPI_BYTE, size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if(status.MPI_TAG == TAG_END) break;
                std::cout << +byte << std::endl;
            }

            return 0;
        }

        // Recieve for all except first proccess
        void recieve()
        {
            while (!lastNum) 
            {
                MPI_Recv(&byte, 1, MPI_BYTE, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                send();

                switch (status.MPI_TAG) 
                {
                    case TAG_TOP:
                        queueTop.push(byte);
                        break;
                    case TAG_BOTTOM:
                        queueBottom.push(byte);
                        break;
                    case TAG_END:
                        lastNum = true;
                        handleSend(TAG_END);
                        return;
                    default:
                        break;
                }
            }
        }

        // Sending for all proccesses
        void handleSend(Tag mode)
        {
            int reciever = (lastNode) ? 0 : rank + 1;   // if last node, reciever is first node

            if(mode == TAG_TOP)
            {
                MPI_Send(&queueTop.front(), 1, MPI_BYTE, reciever, selector, MPI_COMM_WORLD);
                queueTop.pop();
                cntTop--;
            }
            else if(mode == TAG_BOTTOM)
            {
                MPI_Send(&queueBottom.front(), 1, MPI_BYTE, reciever, selector, MPI_COMM_WORLD);
                queueBottom.pop();
                cntBottom--;
            }
            else if(mode == TAG_END)
            {
                lastNum = true;
                while (!queueTop.empty() || !queueBottom.empty()) 
                {
                    send();
                }
                MPI_Send(&byte, 1, MPI_BYTE, reciever, TAG_END, MPI_COMM_WORLD);
            }
        }

        // Sending and ordering logic
        void send()
        {
            // PRINT REST OF SERIES
            if(cntTop == 0)
            {
                handleSend(TAG_BOTTOM);
                switchSelector();
                return;
            }
            else if(cntBottom == 0)
            {
                handleSend(TAG_TOP);
                switchSelector();
                return;
            }

            // STANDART SENDING, at lest one number in queueBottom, queueTop is full
            if(!queueBottom.empty() || (enoughtNumbers && !queueTop.empty() && !queueBottom.empty()) )
            {
                enoughtNumbers = true;
                if(queueBottom.front() < queueTop.front())
                {
                    handleSend(TAG_BOTTOM);
                }
                else
                {
                    handleSend(TAG_TOP);
                }
            }

            // SEND REST AFTER LAST NUMBER RECIEVED
            else if(lastNum && !queueTop.empty())
            {
                handleSend(TAG_TOP);
            }
            else if(lastNum && !queueBottom.empty())
            {
                handleSend(TAG_BOTTOM);
            }
        };

        // Switching recievers queue
        void switchSelector() 
        {
            if(cntTop || cntBottom) return; // If there are still numbers in current series
            cntTop = pow(2, rank - 1);
            cntBottom = pow(2, rank - 1);
            selector = (selector) ? TAG_TOP : TAG_BOTTOM;
            enoughtNumbers = false;
        }
};




int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Node *node = new Node(rank, size);
    if(node->execute())
    {
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}