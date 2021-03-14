//
//  main.cpp
//  Selection_algorithm
//
//  Created by lianxing on 2021/2/1.
//  Copyright © 2021 lianxing. All rights reserved.
//

#include <iostream>
using namespace std;

void selectSort(int r[],int n){
    int i,j;
    float temp;
    for(i=0; i<n; i++)
    {
        for(j=i+1; j<n; j++)
        {
            if(r[i] > r[j])
            {
                temp = r[i];
                r[i] = r[j];
                r[j] = temp;
            }
        }
    }
    
}
void bubble_sort(int r[],int n){
    int i,j;
    float temp;
    for ( i = 0; i < n; i++)
    {
        for(j=1; j<n-i; j++)
        {
            if (r[j-1] > r[j])
            {
                temp = r[j-1];
                r[j-1] = r[j];
                r[j] = temp;
            }
        }
    }
    
}

void insertion_sort(int r[],int n){
    int i;
    float temp;
    for(i=1; i<n; i++){
        while(i>0&&(r[i]<r[i-1]))
        {
            temp = r[i-1];
            r[i-1] = r[i];
            r[i] = temp;
            i -= 1;
        }
        
    }
}

void shell_sort(int r[],int n ){
    int i,temp,gap;
    gap = n/2;
    while (gap)
    {
        for(i=gap;i<n;i++)
        {
            while ((i-gap >=0)&&(r[i] < r[i-gap]))
            {
                temp = r[i];
                r[i] = r[i-gap];
                r[i-gap] = temp;
                i = i-gap;
            }
            
        }
        gap = gap/2;

    }
}


void merge(int *A,int *L,int *R,int L_nums,int R_nums){
    int i,j,k;
    i=0;j=0;k=0;
    while((i<L_nums)&&(j<R_nums))
    {
        if(L[i]<R[j]) A[k++] = L[i++];
        else A[k++] = R[j++];
    }
    while(i<L_nums) A[k++] = L[i++];
    while(j<R_nums) A[k++] = R[j++];
    
}
void merge_sort(int *r,int n){
    

    int mid,i, *L, *R;
	if(n < 2) return; // base condition. If the array has less than two element, do nothing.
 
	mid = n/2;  // find the mid index.
 
	// create left and right subarrays
	// mid elements (from index 0 till mid-1) should be part of left sub-array
	// and (n-mid) elements (from mid to n-1) will be part of right sub-array
	L = new int[mid];
	R = new int [n - mid];
 
	for(i = 0;i<mid;i++) L[i] = r[i]; // creating left subarray
	for(i = mid;i<n;i++) R[i-mid] = r[i]; // creating right subarray

    merge_sort(L,mid);
    merge_sort(R,n-mid);
    merge(r,L,R,mid,n-mid);

    delete [] R;
	delete [] L;


}

void quick_sort(int r[],int n,int left,int right){
    int temp,pivot;
    if (left>=right){
        return;
    }
    pivot = left;
    int i = left;
    int j = right;
    while (i<j) {
        while ((i<j)&&(r[j]>r[pivot])) j--;
        while ((i<j)&&(r[i]<=r[pivot])) i++;
        temp=r[i];r[i]=r[j];r[j]=temp;
    temp=r[i];r[i]=r[pivot];r[pivot]=temp;
    quick_sort(r, n, left, i-1);
    quick_sort(r, n, i+1, right);
    return;
    
    }
        
}








int main()
{
    int r[]  = {0,5,6,8,4,9,6,74,65,123,94};
    int n = sizeof(r)/sizeof(r[0]);
    // int *a = begin(r);
    // int *b = end(r);

    cout << "Original nums:" << endl;
    for(int i=0; i<n; i++)
    {
        cout<<r[i]<<"  ";
    }
    
    int flag_selection = 0;
    int flag_merge = 0;
    int flag_bubble = 0;
    int flag_shell = 0;
    int flag_insertion = 0;
    int flag_quick = 1;
    
    

    // //选择排序
    if (flag_selection == 1)
        { selectSort(r, n);
         cout << "选择排序结果: " << endl;
         for(int i=0; i<n; i++)
         {
             cout<<r[i]<<"  "<<endl;
         }}


     //冒泡排序
    if (flag_bubble == 1)
         {bubble_sort(r,n);
         cout << "Bubble_sort: " << endl;
         for(int i=0; i<n; i++)
         {
             cout<<r[i]<<"  ";
         }}

       //插入排序
    if (flag_insertion == 1)
         {insertion_sort(r,n);
         cout << "Insertion_sort: " << endl;
         for(int i=0; i<n; i++)
         {
             cout<<r[i]<<"  ";
         }}
    
    
     //   希尔排序
    if (flag_shell == 1)
         {shell_sort(r,n);
         cout << "Shell_sort: " << endl;
         for(int i=0; i<n; i++)
         {
             cout<<r[i]<<"  ";
         }}

    //   归并排序
    if (flag_merge == 1)
        {merge_sort(r,n);
        cout << "Merge_sort: " << endl;
        for(int i=0; i<n; i++)
        {
            cout<<r[i]<<"  ";
        }}

    //快速排序
    if (flag_quick == 1)
    {quick_sort(r, n, 0, n-1);
        cout << "Quick_sort:"<<endl;
        for (int i=0; i<n; i++) {
            cout<<r[i]<<" ";
        }
            
        }

    return 0;



}


