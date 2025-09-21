#include <bits/stdc++.h>
using namespace std;
#define SIZE 4096
void allocate_matrix(int ***ptr)
{
    *ptr = (int**)malloc(SIZE * sizeof(int *));
     for(int i = 0 ; i <SIZE ; i++)
     {
        (*ptr)[i] = (int*)malloc(SIZE * sizeof(int));
     }
    
}
void initiate_matrix(int **matrix , int flag)
{
    for(int i = 0 ; i < SIZE ; i++)
    {
        for(int j = 0 ; j < SIZE ; j++)
        {
            if(flag==0)
            matrix[i][j] = rand() % 10;
            else if(flag = 2)
            {
                if(i == j)
                    matrix[i][j] = 1;
                else
                    matrix[i][j] = 0;
            }
            else 
                matrix[i][j] = 0;
        }
    }
}
void multiplication_order1(int **A,int **B,int **C)
{
    for(int i = 0 ; i < SIZE ; i++)
    {
        for(int j = 0 ; j < SIZE ; j++)
        {
            for(int k = 0 ; k < SIZE ; k++)
            {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

int main()
{
    int **A, **B, **C;
    allocate_matrix(&A);
    allocate_matrix(&B);
    allocate_matrix(&C);

    initiate_matrix(A,1);
    initiate_matrix(B,1);
    initiate_matrix(C,0);
    multiplication_order1(A, B, C);
}